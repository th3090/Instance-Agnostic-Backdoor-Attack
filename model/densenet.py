"""
DenseNet in PyTorch

"""

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        # self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        # out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv1(F.relu(x))
        # out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv2(F.relu(out))
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        # self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(x))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, block_config, growth_rate=32, reduction=0.5, num_classes=10, init_weights=True):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate

        self.first_conv = nn.Sequential(
            nn.Conv2d(3, num_planes, kernel_size=7, stride=2, padding=3, bias=False),
            # nn.BatchNorm2d(num_planes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.dense1 = self._make_dense_layers(block, num_planes, block_config[0])
        num_planes += block_config[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, block_config[1])
        num_planes += block_config[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, block_config[2])
        num_planes += block_config[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, block_config[3])
        num_planes += block_config[3]*growth_rate

        # self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

        if init_weights:
            self._initialize_weights()

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def penultimate(self, x):
        out = self.first_conv(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        # out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = F.avg_pool2d(F.relu(out), 4)
        out = out.view(out.size(0), -1)

        return out

    def get_block_feats(self, x):
        feat_list = []

        out = self.first_conv(x)
        feat_list.append(out)

        out = self.trans1(self.dense1(out))
        feat_list.append(out)

        out = self.trans2(self.dense2(out))
        feat_list.append(out)

        out = self.trans3(self.dense3(out))
        feat_list.append(out)

        out = self.dense4(out)
        feat_list.append(out)

        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        feat_list.append(out)

        return feat_list

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

    def pretrain_layer(self, num_classes=10):
        if num_classes != self.linear.out_features:
            self.linear = nn.Linear(in_features=self.linear.in_features, out_features=num_classes, bias=False)

        for name, param in self.named_parameters():
            param.requires_grad = False
            if 'linear' in name:
                param.requires_grad = True


def denseNet121(pretrained: bool = False, check_point: bool = False, progress: bool = True,
                check_point_path=None, num_classes=None, **kwargs: Any):

    model = DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32)
    if pretrained and check_point is False:
        try:
            from torch.hub import load_state_dict_from_url
        except ImportError:
            from torch.utils.model_zoo import load_url as load_state_dict_from_url
        state_dict = load_state_dict_from_url(
            "https://download.pytorch.org/models/densenet121-a639ec97.pth", progress=True)
        model.load_state_dict(state_dict, strict=False)

        return model

    elif pretrained is False and check_point:
        model.linear = nn.Linear(in_features=model.linear.in_features, out_features=num_classes, bias=False)
        model.load_state_dict(torch.load(check_point_path), strict=False)

        return model

    model.linear = nn.Linear(in_features=model.linear.in_features, out_features=num_classes, bias=False)

    return model