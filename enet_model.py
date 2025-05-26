from efficientnet_pytorch import model as enet
import torch.nn as nn
import torch.nn.functional as F
import torch

class L2Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(L2Loss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self._reduction = reduction

    def set_reduction(self, reduction):
        assert reduction in ['none', 'sum', 'mean']
        self._reduction = reduction

    def forward(self, x, y):
        assert len(x.shape) == 4
        losses = ((x - y) * (x - y)).sum(3).sum(2).sum(1) / (x.size(1) * x.size(2) * x.size(3))
        if self._reduction == 'none':
            return losses
        elif self._reduction == 'mean':
            return torch.mean(losses)
        else:
            return torch.sum(losses)

class enetv2(nn.Module):
    def __init__(self, backbone, out_dim):
        self.patch_count = 12 #parameter
    
        super(enetv2, self).__init__()
        self.enet = enet.EfficientNet.from_pretrained(backbone)
        self.myfc = nn.Linear(self.enet._fc.in_features, out_dim)
        self.enet._fc = nn.Identity()

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        feature = self.extract(x)
        
        x = self.myfc(feature)
        return x
        
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = (1-pt)**self.gamma * BCE_loss * self.alpha

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
