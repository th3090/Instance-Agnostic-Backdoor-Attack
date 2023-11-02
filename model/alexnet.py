import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5,
                 init_weights=True) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self.block = nn.Sequential(
            # nn.Dropout(p=dropout)
        )

        self.fc1_extractor = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

        self.penultimate = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )

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

    def forward(self, x, penu=False, fc1_extractor=False, block=False):
        # if block:
        #     return self.get_linear_block(x)

        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        if block:
            # out = self.block(out)
            return out

        if fc1_extractor:
            out = self.fc1_extractor(out)
            return out

        if penu:
            out = self.penultimate(out)
            return out

        out = self.classifier(out)
        return out

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


def alexnet(pretrained=True, check_point=False, check_point_path=None, progress=True, num_classes=None, **kwargs):
    if pretrained or check_point:
        kwargs['init_weights'] = False

    model = AlexNet(**kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(
            "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth", progress=progress)
        model.load_state_dict(state_dict, strict=False)

        return model

    elif pretrained is False and check_point:
        model.classifier[-1] = nn.Linear(in_features=model.classifier[-1].in_features, out_features=num_classes)
        model.load_state_dict(torch.load(check_point_path), strict=False)

        return model

    model.classifier[-1] = nn.Linear(in_features=model.classifier[-1].in_features, out_features=num_classes)

    return model



