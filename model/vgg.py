import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes: int = 1000, dropout: float = 0.5,
                 init_weights=True):

        super(VGG, self).__init__()
        self.num_classes = num_classes

        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )

        self.fc1_extractor = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
        )

        self.penultimate = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
        )

        if init_weights:
            self._initialize_weights()

    def get_linear_block(self, x):
        feat_list = []

        out = self.features(x)
        out = self.avgpool(out)
        feat_list.append(out)

        out = torch.flatten(out, 1)

        y = self.fc1_extractor(out)
        feat_list.append(y)

        y = self.penultimate(out)
        feat_list.append(y)

        # y = self.classifier(out)
        # feat_list.append(y)

        return feat_list

    def get_e2e(self, x):
        feat_list = []

        out = self.features[:5](x)
        feat_list.append(out)

        out = self.features[:9](x)
        feat_list.append(out)

        out = self.features[:16](x)
        feat_list.append(out)

        out = self.features[:23](x)
        feat_list.append(out)

        out = self.features(x)
        out = self.avgpool(out)
        feat_list.append(out)

        out = torch.flatten(out, 1)

        y = self.fc1_extractor(out)
        feat_list.append(y)

        y = self.penultimate(out)
        feat_list.append(y)

        # y = self.classifier(out)
        # feat_list.append(y)

        return feat_list

    def forward(self, x, fc1_extractor=False, penu=False, block=False):
        # if block:
        #     return self.get_linear_block(x)

        # if block:
        #     return self.get_e2e(x)

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if block:
            output = x
            return output

        if fc1_extractor:
            output = self.fc1_extractor(x)
            return output

        if penu:
            output = self.penultimate(x)
            return output

        output = self.classifier(x)

        return output

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

    def pretrain_layer(self, num_classes=10, classifier_target_layer=6):
        if num_classes != self.classifier[-1].out_features:
            self.classifier[-1] = nn.Linear(in_features=self.classifier[-1].in_features, out_features=num_classes,
                                            bias=True)

        for name, param in self.named_parameters():
            param.requires_grad = False

        for param in self.classifier[classifier_target_layer:].parameters():
            param.requires_grad = True


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, check_point, check_point_path, progress, **kwargs):
    if pretrained or check_point:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict, strict=False)
        return model

    if check_point:
        model.load_state_dict(torch.load(check_point_path))
        return model

    return model


"""
VGG 11-layer models (configuration "A")
VGG 13-layer models (configuration "B")
VGG 16-layer models (configuration "D")
VGG 19-layer models (configuration "E")
Args:
    pretrained (bool): If True, returns a models pretrained on ImageNet
    progress (bool): If True, displays a progress bar of the download to stderr
"""


def vgg11(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)


def vgg11_bn(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)


def vgg13(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)


def vgg16(pretrained=False, check_point=False, check_point_path=None, progress=True, **kwargs):
    return _vgg('vgg16', 'D', False, pretrained, check_point, check_point_path, progress, **kwargs)


def vgg16_bn(pretrained=False, check_point=False, check_point_path=None, progress=True, **kwargs):
    return _vgg('vgg16_bn', 'D', True, pretrained, check_point, check_point_path, progress, **kwargs)


def vgg19(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)
