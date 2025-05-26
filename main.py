import os
import argparse
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from pytorch.unet_student import *
from pytorch.unet_teacher import *
from pytorch.unet_concat import *
from enet_model import FocalLoss, L2Loss
from sklearn.metrics import confusion_matrix, roc_auc_score
import numpy as np
from dataset import *
from torch.autograd import Variable
from collections import defaultdict, OrderedDict
import torch.nn.functional as F

# Set random seed to make the experiment reproducible
def seed_torch(seed=1029):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Get the ViT-B/16 configuration
def get_b16_config():
    import ml_collections
    config = ml_collections.ConfigDict()
    config.hidden_size = 512
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 512 * 4
    config.transformer.num_heads = 4
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config

# Set arguments and hyper-parameters
parser = argparse.ArgumentParser(description='MIX_train')
parser.add_argument('--data_dir', default='/home/hcc/', type=str, help='Directory to load PCam data.')
parser.add_argument('--label_dir', default='./label', type=str, help='Directory to load label files.')
parser.add_argument('--ckpt_dir', default='./hcc_model/', type=str, help='Directory to save checkpoint.')
parser.add_argument('--sequence', nargs='+', default='arterial', type=str, help='Directory to load label files.')
parser.add_argument('--gpu', default='1', type=str, help='GPU Devices to use.')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size.')
parser.add_argument('--lr', default=3e-4, type=float, help='Starting learning rate.')
parser.add_argument('--lr_decay', default=0.9, type=float, help='Learning rate decay.')
parser.add_argument('--lr_decay_step', default=2, type=int, help='Learning rate decay step.')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='L2 penalty for regularization.')
parser.add_argument('--start_epoch', default=1, type=int, help='Starting epoch.')
parser.add_argument('--epochs', default=20, type=int, help='Number of training epochs.')
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint or not.')
parser.add_argument('--store_last', action='store_true', help='store the last model.')
parser.add_argument('--resume_last', action='store_true', help='resume the last model.')
parser.add_argument('--name', default='Stu_distilling', type=str, help='The id of this train')
parser.add_argument('--valid_fold', default=0, type=int, help='Starting epoch.')
parser.add_argument('--seeds', default=0, type=int, help='Starting epoch.')
args = parser.parse_args()

# Positional Encoding class
class PositionalEncoding(nn.Module):
    def __init__(self, length = 52):
        super(PositionalEncoding, self).__init__()
        self.weights = nn.Parameter(torch.ones(1, length))
        self.m = nn.ReLU()

    def forward(self, x):
        weight_f = self.m(self.weights / torch.max(self.weights))
        embedding = x * weight_f
        return embedding

# Set random seed
seed_torch(args.seeds)

# Create checkpoint directory if it doesn't exist
if not os.path.exists(args.ckpt_dir):
    os.makedirs(args.ckpt_dir)
print('==> Arguments:')
print(args)

# Set GPU device
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize best accuracy and output dimension
best_acc = 0
out_dim = 3

# Load teacher model
teacher = concat_teacherNet(num_classes=out_dim, num_sequ=1, config=get_b16_config()).to(device)
teacher.train(mode=False)

# Load student model
student = concat_studentNet(num_classes=out_dim, num_sequ=1, config=get_b16_config()).to(device)
student.train(mode=False)

# Initialize positional encoding
pos_emb = PositionalEncoding().to(device)

# Initialize cfeature prediction model
cfeature_predict = predict_cfeature(num_sequ=1).to(device)

# Define loss functions
lossKD = nn.KLDivLoss()
lossBCE = torch.nn.BCEWithLogitsLoss()
lossCE = FocalLoss()
L2loss = L2Loss()
model_num = 2

# Define optimizers and schedulers
optimizer_stu = torch.optim.Adam(filter(lambda p: p.requires_grad, student.parameters()), lr=args.lr)
scheduler_stu = torch.optim.lr_scheduler.StepLR(optimizer_stu, args.lr_decay_step, gamma=args.lr_decay)
optimizer_lw = torch.optim.Adam(filter(lambda p: p.requires_grad, pos_emb.parameters()), lr=0.05)
optimizer_tea = torch.optim.Adam(filter(lambda p: p.requires_grad, teacher.parameters()), lr=args.lr)
scheduler_tea = torch.optim.lr_scheduler.StepLR(optimizer_tea, args.lr_decay_step, gamma=args.lr_decay)

optimizers = []
optimizers.append(optimizer_stu)
optimizers.append(optimizer_tea)

# Load data from dataset
def load_data(data_dir, label_dir):
    """
    Load data from dataset.

    # Arguments
        data_dir (str): Directory to load data.
        label_dir (str): Directory to load label files.

    # Returns
        Pytorch dataloader for each set.
    """
    train_set = hcc_3d_all(data_dir, label_dir, usage='train', valid_fold=args.valid_fold, sequ_name=args.sequence)
    valid_set = hcc_3d_all(data_dir, label_dir, usage='valid', valid_fold=args.valid_fold, sequ_name=args.sequence)

    train_label = train_set.get_target()
    count = [0] * 3
    for l in train_label:
        count[l] += 1
    count[0] = count[0] * 5

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
    return train_loader, valid_loader

# Train the model with training set
def train(epoch, train_loader):
    """
    Train model with training set.

    # Arguments:
        epoch (int): Current epoch
        train_loader: Pytorch dataloader for train set.
    """
    print("Training")
    student.train()
    teacher.train()

    TARGETS = []
    PREDS = []
    PREDS_tea = []
    losses = []
    losses_tea = []

    l2 = L2Loss(reduction='none')

    for idx, (data, target, mask, cfeature, new_cfeature) in enumerate(train_loader):
        losses_ = []
        data = data.to(device)
        target = target.float().to(device)

        hbp = data[:, 0:1, :, :, :]
        pre = data[:, 1:2, :, :, :]

        mask = mask.float().to(device)
        hbp_mask = mask[:, 0:1, :, :, :]
        pre_mask = mask[:, 1:2, :, :, :]

        cfeature = cfeature.float().to(device)
        new_cfeature = new_cfeature.float().to(device)

        teacher_final_logits, teacher_cls_token, teacher_pre, teacher_hbp = teacher(hbp, pre, hbp_mask, pre_mask, cfeature)
        student_final_logits, student_cls_token, student_pre, student_hbp = student(hbp, pre, hbp_mask, pre_mask)
        cfeature_logits = cfeature_predict(student_cls_token).float().to(device)

        outputs = []
        outputs.append(student_final_logits)
        outputs.append(teacher_final_logits)

        optimizer_lw.zero_grad()
        for i in range(model_num):
            optimizers[i].zero_grad()
            loss_ce = lossCE(outputs[i], target)

            if i == 0 and not args.no_kd:
                loss_kd_hbp = lossKD(F.log_softmax(student_hbp, dim=1), F.softmax(Variable(teacher_hbp), dim=1))
                loss_kd_pre = lossKD(F.log_softmax(student_pre, dim=1), F.softmax(Variable(teacher_pre), dim=1))
                loss_kd = lossKD(F.log_softmax(outputs[0], dim=1), F.softmax(Variable(outputs[1]), dim=1))
                loss_cfeature = l2(cfeature_logits, cfeature)
                embeddings = pos_emb(loss_cfeature)
                update_loss_cfeature = torch.mean(embeddings)
                loss = loss_ce + loss_kd + update_loss_cfeature + loss_kd_hbp + loss_kd_pre
            else:
                loss = loss_ce

            losses_.append(loss)
            loss.backward()
            optimizers[i].step()

        optimizer_lw.step()

        y_student = outputs[0]
        pred_stu = torch.squeeze(y_student.max(1)[1])
        PREDS.append(pred_stu)

        y_teacher = outputs[1]
        pred_tea = torch.squeeze(y_teacher.max(1)[1])
        PREDS_tea.append(pred_tea)

        TARGETS.append(target.max(1)[1])

        acc_stu = (pred_stu == target.max(1)[1]).cpu().numpy().mean()
        loss_np_stu = losses_[0].detach().cpu().numpy()
        losses.append(loss_np_stu)
        smooth_loss_stu = sum(losses[-100:]) / min(len(losses), 100)

        acc_tea = (pred_tea == target.max(1)[1]).cpu().numpy().mean()
        loss_np_tea = losses_[1].detach().cpu().numpy()
        losses_tea.append(loss_np_tea)
        smooth_loss_tea = sum(losses_tea[-100:]) / min(len(losses_tea), 100)

        print(idx, smooth_loss_stu, acc_stu, end="\r")
        print(idx, smooth_loss_tea, acc_tea, end="\r")

    print(pos_emb.weights)
    print()

    PREDS = torch.cat(PREDS).cpu().numpy()
    PREDS_tea = torch.cat(PREDS_tea).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()

    get_confusion_matrix(3, PREDS, TARGETS)
    get_confusion_matrix(3, PREDS_tea, TARGETS)

    scheduler_stu.step()
    scheduler_tea.step()
    return losses, losses_tea

# Evaluate the model on the validation set
def eval_training(epoch, valid_loader):
    global best_acc
    student.eval()
    teacher.eval()

    val_loss = []
    LOGITS = []
    PREDS = []
    val_loss1 = []
    LOGITS1 = []
    PREDS1 = []
    TARGETS = []
    TARGETS_OH = []

    with torch.no_grad():
        for idx, (data, target, mask, cfeature, new_cfeature) in enumerate(valid_loader):
            print(idx, end='\r')
            data, target = data.to(device), target.float().to(device)
            cfeature = cfeature.float().to(device)

            hbp = data[:, 0:1, :, :, :]
            pre = data[:, 1:2, :, :, :]

            mask = mask.float().to(device)
            hbp_mask = mask[:, 0:1, :, :, :]
            pre_mask = mask[:, 1:2, :, :, :]

            teacher_final_logits, teacher_cls_token, teacher_pre, teacher_hbp = teacher(hbp, pre, hbp_mask, pre_mask, cfeature)
            student_final_logits, student_cls_token, student_pre, student_hbp = student(hbp, pre, hbp_mask, pre_mask)
            cfeature_logits = cfeature_predict(student_cls_token).float().to(device)

            loss = lossCE(student_final_logits, target)
            loss1 = lossCE(teacher_final_logits, target)

            pred = student_final_logits.sigmoid().max(1)[1]
            LOGITS.append(student_final_logits)
            PREDS.append(pred)

            pred1 = teacher_final_logits.sigmoid().max(1)[1]
            LOGITS1.append(teacher_final_logits)
            PREDS1.append(pred1)

            TARGETS.append(target.max(1)[1])
            TARGETS_OH.append(target)

            val_loss.append(loss.detach().cpu().numpy())
            val_loss1.append(loss1.detach().cpu().numpy())

        val_loss = np.mean(val_loss)
        val_loss1 = np.mean(val_loss1)

    LOGITS = torch.cat(LOGITS).cpu()
    LOGITS1 = torch.cat(LOGITS1).cpu()

    LOGITS2 = F.softmax(LOGITS, 1).numpy()
    LOGITS2_ = F.softmax(LOGITS1, 1).numpy()

    PREDS = torch.cat(PREDS).cpu().numpy()
    PREDS1 = torch.cat(PREDS1).cpu().numpy()

    TARGETS = torch.cat(TARGETS).cpu().numpy()
    TARGETS_OH = torch.cat(TARGETS_OH).cpu().numpy()

    acc = (PREDS == TARGETS).mean() * 100.
    auc_case = roc_auc_score(np.round(np.array(TARGETS_OH), 0), np.array(LOGITS2), average="macro", multi_class="ovo")
    auc_case2 = roc_auc_score(np.round(np.array(TARGETS_OH), 0), np.array(LOGITS), average="macro", multi_class="ovo")

    acc1 = (PREDS1 == TARGETS).mean() * 100.
    auc_case1 = roc_auc_score(np.round(np.array(TARGETS_OH), 0), np.array(LOGITS2_), average="macro", multi_class="ovo")
    auc_case2_ = roc_auc_score(np.round(np.array(TARGETS_OH), 0), np.array(LOGITS1), average="macro", multi_class="ovo")

    print("----------------------------student_results-------------------------------")
    get_confusion_matrix(3, PREDS, TARGETS)
    print(acc, val_loss, auc_case, auc_case2)

    print()

    print("----------------------------teacher_results-------------------------------")
    get_confusion_matrix(3, PREDS1, TARGETS)
    print(acc1, val_loss1, auc_case1, auc_case2_)

    if auc_case > best_acc:
        print("************************")
        best_acc = auc_case
        save_checkpoint(epoch)

# Calculate and print the confusion matrix
def get_confusion_matrix(n_classes, pred, target):
    confusion_matrix = np.zeros((n_classes, n_classes))
    for p, t in zip(pred, target):
        confusion_matrix[int(p), int(t)] += 1
    print(confusion_matrix)

# Save the checkpoint if the accuracy is higher than before
def save_checkpoint(epoch, name=None):
    """
    Save checkpoint if accuracy is higher than before.

    # Arguments
        epoch (int): Current epoch.
    """
    global best_acc
    print('==> Saving checkpoint...')
    state = {
        'model': student,
        'teacher': teacher,
        'epoch': epoch,
        'acc': best_acc,
        'weight': pos_emb,
    }
    if name is None:
        checkpoint_name = "ours" + "_" + args.sequence[0] + str(epoch).zfill(3) + '_fold' + str(args.valid_fold) + '%.t7'
    else:
        checkpoint_name = "ours" + "_" + name + ".t7"
    torch.save(state, os.path.join(args.ckpt_dir, checkpoint_name))

if __name__ == '__main__':
    train_loader, valid_loader = load_data(args.data_dir, args.label_dir)
    for epoch in range(args.start_epoch, args.epochs + 1):
        print('\n************** Epoch: %d **************' % epoch)
        train(epoch, train_loader)
        print()
        print()
        eval_training(epoch, valid_loader)