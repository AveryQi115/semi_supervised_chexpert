import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import types
from functools import partial
from resnest.torch import resnest101, resnest50, resnest200

def forward(self, x):
    # Patch function
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    return x

def resnest50_b(cfg, pretrain=True, pretrained_model=None, last_layer_stride=1):
    model = resnest50(pretrained=False)
    model.forward = types.MethodType(forward, model)
    model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
    return model

def resnest101_b(cfg, pretrain=True, pretrained_model=None, last_layer_stride=1):
    model = resnest101(pretrained=False)
    model.forward = types.MethodType(forward, model)
    model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
    return model

def resnest200_b(cfg, pretrain=True, pretrained_model=None, last_layer_stride=1):
    model = resnest200(pretrained=False)
    model.forward = types.MethodType(forward, model)
    model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
    return model
