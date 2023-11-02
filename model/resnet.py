"""
ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

ResNet 50 / 101/ 152
"""
# import math
from typing import Any

import torch
import torch.nn as nn
# import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))

        x = self.relu(self.batch_norm2(self.conv2(x)))

        x = self.conv3(x)
        x = self.batch_norm3(x)

        # downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        # add identity
        x += identity
        x = self.relu(x)

        return x


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.batch_norm2(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        print(x.shape)
        print(identity.shape)
        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):

    def __init__(self, ResBlock, layer_list, num_classes=10, num_channels=3, middle_feat_num=1, init_weights=True):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.middle_feat_num = middle_feat_num

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * ResBlock.expansion, num_classes)

        if init_weights:
            self._initialize_weights()

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * ResBlock.expansion)
            )

        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion

        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)

    def penultimate(self, x):
        out = self.relu(self.batch_norm1(self.conv1(x)))
        out = self.max_pool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x, penu=False, block=False):
        if block:
            return self.get_block_feats(x)

        out = self.penultimate(x)
        if penu:
            return out

        out = self.linear(out)
        return out

    def get_penultimate_params_list(self):
        return [param for name, param in self.named_parameters() if 'linear' in name]

    def reset_last_layer(self):
        self.linear.weight.data.normal_(0, 0.1)
        self.linear.bias.data.zero_()

    def get_block_feats(self, x):
        feat_list = []

        out = self.relu(self.batch_norm1(self.conv1(x)))
        out = self.max_pool(out)
        feat_list.append(out)

        out = self.layer1(out)
        feat_list.append(out)

        out = self.layer2(out)
        feat_list.append(out)

        out = self.layer3(out)
        feat_list.append(out)

        # out = self.layer4(out)
        for nl, layer in enumerate(self.layer4):
            out = layer(out)
            if self.middle_feat_num >= len(self.layer4) - nl - 1 > 0:
                feat_list.append(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        feat_list.append(out)

        return feat_list

    def pretrain_layer(self, num_classes=10, target_block=0):
        if num_classes != self.linear.out_features:
            self.linear = nn.Linear(in_features=self.linear.in_features, out_features=num_classes, bias=False)

        for name, param in self.named_parameters():
            param.requires_grad = False
            if 'linear' in name:
                param.requires_grad = True

        if target_block == 4:
            for param in self.layer4.parameters():
                param.requires_grad = True

        elif target_block == 3:
            for param in self.layer3.parameters():
                param.requires_grad = True
            for param in self.layer4.parameters():
                param.requires_grad = True

        elif target_block == 2:
            for param in self.layer2.parameters():
                param.requires_grad = True
            for param in self.layer3.parameters():
                param.requires_grad = True
            for param in self.layer4.parameters():
                param.requires_grad = True

        elif target_block == 1:
            for param in self.layer1.parameters():
                param.requires_grad = True
            for param in self.layer2.parameters():
                param.requires_grad = True
            for param in self.layer3.parameters():
                param.requires_grad = True
            for param in self.layer4.parameters():
                param.requires_grad = True

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)


def resNet50(pretrained: bool = False, check_point: bool = False,
             check_point_path=None, num_classes=None, **kwargs: Any):

    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained and check_point is False:
        try:
            from torch.hub import load_state_dict_from_url
        except ImportError:
            from torch.utils.model_zoo import load_url as load_state_dict_from_url
        state_dict = load_state_dict_from_url(
            'https://download.pytorch.org/models/resnet50-19c8e357.pth', progress=True)
        model.load_state_dict(state_dict, strict=False)

        return model

    elif pretrained is False and check_point:
        model.linear = nn.Linear(in_features=model.linear.in_features, out_features=41, bias=False)
        model.load_state_dict(torch.load(check_point_path), strict=False)

        return model

    model.linear = nn.Linear(in_features=model.linear.in_features, out_features=num_classes, bias=False)

    return model
