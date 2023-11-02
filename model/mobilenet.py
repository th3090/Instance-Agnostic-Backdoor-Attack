"""
MobileNetV2 in PyTorch.
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.

pytorch pretrained weight = https://gaussian37.github.io/dl-concept-mobilenet_v2/ 참고
"""
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    # expand + depthwise + pointwise

    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride
        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1, 16, 1, 1),
           (6, 24, 2, 2),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6, 32, 3, 2),
           (6, 64, 4, 2),
           (6, 96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10, init_weights=True):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

        if init_weights:
            self._initialize_weights()

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for stri in strides:
                layers.append(Block(in_planes, out_planes, expansion, stri))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def penultimate(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x, penu=False):

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

    def pretrain_layer(self, num_classes=10, n_block_layer=False):
        if num_classes != self.linear.out_features:
            self.linear = nn.Linear(in_features=self.linear.in_features, out_features=num_classes, bias=False)

        for name, param in self.named_parameters():
            param.requires_grad = False
            if 'linear' in name:
                param.requires_grad = True

        if not n_block_layer:
            for param in self.layers[-n_block_layer:].parameters():
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


def mobilenetV2(pretrained: bool = False, check_point: bool = False,
                check_point_path=None, num_classes=None,
                progress=True, **kwargs: Any) -> MobileNetV2:
    """
    Constructs a MobileNetV2 architecture from MobileNetV2
    :param num_classes: Correct num_classes to load check_point
    :param check_point_path: user's pretrained weights' path
    :param check_point: If True, use user's pretrained weights
    :param pretrained: If True, returns a model pretrained on ImageNet
    :param progress: If True, displays a progress bar of the download to stderr
    :param kwargs: keyword arguments
    :return: MobileNetV2
    """
    model = MobileNetV2(**kwargs)
    if pretrained and check_point is False:
        try:
            from torch.hub import load_state_dict_from_url
        except ImportError:
            from torch.utils.model_zoo import load_url as load_state_dict_from_url
        state_dict = load_state_dict_from_url(
            'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth', progress=True)
        model.load_state_dict(state_dict, strict=False)

        return model

    elif pretrained is False and check_point:
        model.linear = nn.Linear(in_features=model.linear.in_features, out_features=41, bias=False)
        model.load_state_dict(torch.load(check_point_path), strict=False)

        return model

    model.linear = nn.Linear(in_features=model.linear.in_features, out_features=num_classes, bias=False)

    return model
