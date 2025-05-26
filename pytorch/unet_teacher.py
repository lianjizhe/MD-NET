import torch
import torch.nn as nn
import torch.nn.functional as F
 
class targetNetc(nn.Module):
    def __init__(self, num_classes=3, num_sequ=1):
        super(targetNetc, self).__init__()
        
        self.num_sequ = num_sequ
        self.base_model = UNet3D(num_sequ=num_sequ)
        
        self.dense = nn.Linear(512*2*self.num_sequ, num_classes, bias=True)
        self.dense_1 = nn.Linear(512*2*self.num_sequ, 512*self.num_sequ, bias=True)


    def forward(self, x, mask, cfeature):
        bs = x.shape[0]
        
        out,out_features = self.base_model(x, mask, cfeature)

        out_glb_max_pool = F.max_pool3d(out, kernel_size=out.size()[2:]).view(out.size()[0],-1)
        out_glb_avg_pool = F.avg_pool3d(out, kernel_size=out.size()[2:]).view(out.size()[0],-1)
        
        out = torch.cat([out_glb_avg_pool, out_glb_max_pool], 1).squeeze()
        
        if self.num_sequ != 1:
            out = out.reshape(bs, self.num_sequ, out.shape[1])
            out = out.reshape(bs, self.num_sequ * out.shape[2])
        linear_out = self.dense_1(out)
      
        return linear_out
        
        
class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):

        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, act):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(out_chan)
        if act == 'relu':
            self.activation = nn.ReLU(out_chan)
        elif act == 'prelu':
            self.activation = nn.PReLU(out_chan)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        return out


def _make_nConv(in_channel, depth, act, double_chnnel=False):
    if double_chnnel:
        layer1 = LUConv(in_channel, 32 * (2 ** (depth+1)),act)
        layer2 = LUConv(32 * (2 ** (depth+1)), 32 * (2 ** (depth+1)),act)
    else:
        layer1 = LUConv(in_channel, 32*(2**depth),act)
        layer2 = LUConv(32*(2**depth), 32*(2**depth)*2,act)

    return nn.Sequential(layer1,layer2)

class DownTransition(nn.Module):
    def __init__(self, in_channel,depth, act):
        super(DownTransition, self).__init__()
        self.ops = _make_nConv(in_channel, depth,act)
        self.maxpool = nn.MaxPool3d(2)
        self.current_depth = depth

    def forward(self, x):
        if self.current_depth == 3:
            out = self.ops(x)
            out_before_pool = out
        else:
            out_before_pool = self.ops(x)
            out = self.maxpool(out_before_pool)
        return out, out_before_pool

class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, depth,act):
        super(UpTransition, self).__init__()
        self.depth = depth
        self.up_conv = nn.ConvTranspose3d(inChans, outChans, kernel_size=2, stride=2)
        self.ops = _make_nConv(inChans+ outChans//2,depth, act, double_chnnel=True)

    def forward(self, x, skip_x):
        out_up_conv = self.up_conv(x)
        concat = torch.cat((out_up_conv,skip_x),1)
        out = self.ops(concat)
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, n_labels):

        super(OutputTransition, self).__init__()
        self.final_conv = nn.Conv3d(inChans, n_labels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.final_conv(x))
        return out

class MaskAttentionBlock(nn.Module):
    def __init__(self, inplanes):
        super(MaskAttentionBlock, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.inplanes = inplanes
        
    def forward(self, x, mask):
        out = mask * x
        return out

class ChannelAttentionBlock(nn.Module):
    def __init__(self, inplanes):
        super(ChannelAttentionBlock, self).__init__()
        self.inplanes = inplanes
        
        self.pooling = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(self.inplanes, int(self.inplanes / 16))
        self.fc2 = nn.Linear(int(self.inplanes / 16), self.inplanes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x):
        
        channel_dis = self.pooling(x)
        channel_dis = channel_dis.view(channel_dis.shape[0], -1)
        out = self.relu(self.fc1(channel_dis))
        
        out = self.sigmoid(self.fc2(out)).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        out = (1+out) * x
        
        return out
        
class UNet3D(nn.Module):
    def __init__(self, n_class=1, act='relu', num_sequ=1):
        super(UNet3D, self).__init__()
        self.num_sequ = num_sequ
        self.down_tr64 = DownTransition(1,0,act)
        self.down_tr128 = DownTransition(64,1,act)
        self.down_tr256 = DownTransition(128,2,act)
        self.down_tr512 = DownTransition(256,3,act)
        self.dense_64 = nn.Linear(52, 64, bias=True)
        self.dense_128 = nn.Linear(64, 128, bias=True)
        self.dense_256 = nn.Linear(128, 256, bias=True)
        self.dense_512 = nn.Linear(256, 512, bias=True)
        self.attention0 = MaskAttentionBlock(1)
        mid_channel = 512

        
    def forward(self, x, mask, cfeature):
        feature_maps = []
        out64, _ = self.down_tr64(x)
        cf64 = self.dense_64(cfeature)
        out64 = out64 * cf64.unsqueeze(2).unsqueeze(2).unsqueeze(2)
        feature_maps.append(out64)
        out128, _ = self.down_tr128(out64)
        cf128 = self.dense_128(cf64)
        out128 = out128 * cf128.unsqueeze(2).unsqueeze(2).unsqueeze(2)
        feature_maps.append(out128)
        out256, _ = self.down_tr256(out128)    
        cf256 = self.dense_256(cf128)
        out256 = out256 * cf256.unsqueeze(2).unsqueeze(2).unsqueeze(2)
        feature_maps.append(out256)
        out512, _ = self.down_tr512(out256)
        cf512 = self.dense_512(cf256)
        out512 = out512 * cf512.unsqueeze(2).unsqueeze(2).unsqueeze(2)
        feature_maps.append(out512)
        return out512,feature_maps