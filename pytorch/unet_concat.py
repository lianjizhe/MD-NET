import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from pytorch.unet_student import *
from pytorch.unet_teacher import *
from torch.nn.modules.utils import _pair
from scipy import ndimage

# from .modeling_resnet import ResNetV2
import argparse
import ml_collections

from .transformer import VisionTransformer
from .transformer import *
import pytorch.transformer as transformer
from PIL import Image
import numpy as np

def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.hidden_size = 512
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 512*4
    config.transformer.num_heads = 4
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


class concat_studentNet(nn.Module):
    def __init__(self,num_classes=3,num_sequ=1,config=get_b16_config()):
        super(concat_studentNet, self).__init__()
        self.num_sequ = num_sequ
        self.num_classes = num_classes
        self.pre_model = targetNet(self.num_classes, self.num_sequ)
        self.hbp_model = targetNet(self.num_classes, self.num_sequ)
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(1024, 3, bias=True)
        self.config = config
        self.model_v = VisionTransformer_lh(self.config)

    def forward(self, hbp_data, pre_data, xx, xxx):
        pre_model = self.pre_model(pre_data, return_feature = True)
        hbp_model = self.hbp_model(hbp_data, return_feature = True)
        pre_model_new = torch.unsqueeze(pre_model,dim=1)
        hbp_model_new = torch.unsqueeze(hbp_model,dim=1)
        logits = torch.cat((pre_model_new,hbp_model_new), 1)

        result,_ = self.model_v(logits.to(device)) # b,3,512
        pre_logits = torch.squeeze(result[:,0,:],dim=1)
        hbp_logits = torch.squeeze(result[:,1,:],dim=1)

        cls_pre_hbp_logits = torch.cat((pre_logits,hbp_logits), 1)

        self.linear_out = self.dense(cls_pre_hbp_logits)

        return self.linear_out,cls_pre_hbp_logits,pre_logits,hbp_logits


class predict_cfeature(nn.Module):
    def __init__(self,num_sequ=1):
        super(predict_cfeature, self).__init__()
        self.num_sequ = num_sequ
        self.dropout = nn.Dropout(0.1)
        self.pred_cfeature = nn.Linear(1024, 52, bias=True)
    def forward(self, logits):
        self.cfeature_logits = self.pred_cfeature(logits)
        return self.cfeature_logits

class concat_teacherNet(nn.Module):
    def __init__(self, num_classes=3, num_sequ=1,config=get_b16_config()):
        super(concat_teacherNet, self).__init__()
        self.num_sequ = num_sequ
        self.num_classes = num_classes
        self.pre_model = targetNetc(self.num_classes, self.num_sequ).to(device)
        self.hbp_model = targetNetc(self.num_classes, self.num_sequ).to(device)
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(1024, 3, bias=True)
        self.config = config
        self.model_v = VisionTransformer_lh(self.config)

    def forward(self, hbp_data, pre_data, hbp_mask, pre_mask, cfeature):
        pre_out_features = self.pre_model(pre_data,pre_mask,cfeature)
        hbp_out_features = self.hbp_model(hbp_data,hbp_mask,cfeature)
        pre_model_new = torch.unsqueeze(pre_out_features ,dim=1)
        hbp_model_new = torch.unsqueeze(hbp_out_features,dim=1)
        logits = torch.cat((pre_model_new,hbp_model_new), 1)
        result,_ = self.model_v(logits.to(device)) # b,3,512
        pre_logits = torch.squeeze(result[:,0,:],dim=1)
        hbp_logits = torch.squeeze(result[:,1,:],dim=1)
        cls_pre_hbp_logits = torch.cat((pre_logits,hbp_logits), 1)
        self.linear_out = self.dense(cls_pre_hbp_logits)
        return self.linear_out,cls_pre_hbp_logits,pre_logits,hbp_logits