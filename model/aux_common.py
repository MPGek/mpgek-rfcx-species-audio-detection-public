from typing import List

import timm
import torch
import torch.nn.functional as F
from torch import nn


# Source: https://github.com/luuuyi/CBAM.PyTorch
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


# Source: https://github.com/luuuyi/CBAM.PyTorch
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ConvolutionalBlockAttentionModule(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(ConvolutionalBlockAttentionModule, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, input):
        out = self.ca(input) * input
        out = self.sa(out) * out
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.avg_pool2d(x, 2)
        return x


class SkipBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(SkipBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        if self.scale_factor >= 2:
            x = F.avg_pool2d(x, self.scale_factor)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class AuxBlock(nn.Module):
    def __init__(self, last_fc, num_classes, base_size, dropout):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_size * 8, base_size * last_fc),
            nn.PReLU(),
            nn.BatchNorm1d(base_size * last_fc),
            nn.Dropout(dropout / 2),
            nn.Linear(base_size * last_fc, num_classes),
        )

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class HyperColumnWithShape(nn.Module):
    def __init__(self, skip_target_size, mode="bilinear", align_corners=False):
        super().__init__()
        self.skip_target_size = skip_target_size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, features: List[torch.Tensor]):  # skipcq: PYL-W0221
        layers = []
        dst_size = features[self.skip_target_size].size()[-2:]

        for f in features:
            layers.append(F.interpolate(f, size=dst_size, mode=self.mode, align_corners=self.align_corners))

        return torch.cat(layers, dim=1)


class AuxCommon(nn.Module):
    def __init__(self, backbone, model_top, num_classes, in_channels=3, base_size=64,
                 dropout=0.2, ratio=16, last_filters=8, last_fc=2, pretrained=True, input_height=0):
        super().__init__()

        model_extensions = model_top.split('-')
        model_top = model_extensions[0]
        model_extensions = model_extensions[1:]
        self.model_extensions = model_extensions

        # features backbone
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0,
                                          in_chans=in_channels, features_only=True)
        self.backbone_feature_channels = self.backbone.feature_info.channels()
        backbone_out_features = self.backbone_feature_channels[-1]

        # Skip connections in the Hyper Column with custom size
        self.hyper = None
        present_skip_types = set([f'skip{i}' for i in range(1, 6)]).intersection(model_extensions)
        if present_skip_types:
            skip_type = present_skip_types.pop()
            self.hyper = HyperColumnWithShape(skip_target_size=int(skip_type[-1]) - 1)
            backbone_out_features = sum(self.backbone_feature_channels)

        # Attention modules
        self.pre_attention = None
        if 'pa3' in self.model_extensions or 'pa7' in self.model_extensions:
            pre_attention_kernel_size = 3 if 'pa3' in self.model_extensions else 7
            self.pre_attention = ConvolutionalBlockAttentionModule(backbone_out_features, ratio=ratio,
                                                                   kernel_size=pre_attention_kernel_size)

        self.dilated_conv = None
        if 'dc5' in self.model_extensions:
            self.dilated_conv = nn.Sequential(
                nn.Conv2d(in_channels=backbone_out_features, out_channels=backbone_out_features,
                          kernel_size=5, stride=3, padding=2, dilation=2, bias=True),
                nn.BatchNorm2d(backbone_out_features, eps=0.001),
                nn.SiLU(inplace=True)
            )

        # conv head
        self.conv_head = None
        if model_top != 'fc3':
            self.conv_head = nn.Conv2d(in_channels=backbone_out_features, out_channels=backbone_out_features * 4,
                                       kernel_size=1, stride=1, bias=False)
            self.bn_head = nn.BatchNorm2d(backbone_out_features * 4, eps=0.001)
            self.silu = nn.SiLU(inplace=True)

        self.attention = None
        if 'att' in self.model_extensions:
            self.attention = ConvolutionalBlockAttentionModule(backbone_out_features * 4, ratio=ratio,
                                                               kernel_size=3)

        # Pooling
        self.avg_pool = None
        if model_top != 'fc3':
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Fully Connected modules
        if model_top == 'fc1':
            self.fc = nn.Linear(backbone_out_features * 4, out_features=num_classes)
        elif model_top == 'fc2':
            self.fc = nn.Sequential(
                # nn.Dropout(dropout, inplace=True),
                nn.Linear(backbone_out_features * 4, backbone_out_features // 2),
                nn.PReLU(),
                nn.BatchNorm1d(backbone_out_features // 2),
                # nn.Dropout(dropout / 2, inplace=True),
                nn.Linear(backbone_out_features // 2, num_classes),
            )
        elif model_top == 'fc3':
            target_height = input_height
            for _ in range(5):
                target_height = round(target_height / 2)

            self.fc = nn.Sequential(
                nn.Conv2d(in_channels=backbone_out_features, out_channels=backbone_out_features * 4,
                          kernel_size=(3, 7), stride=1, padding=(1,0), bias=False),
                nn.BatchNorm2d(backbone_out_features * 4, eps=0.001),
                nn.SiLU(inplace=True),
                nn.AdaptiveAvgPool2d((None, 1)),
                nn.Flatten(1),

                nn.Dropout(dropout, inplace=True),
                nn.Linear(backbone_out_features * 4 * target_height, backbone_out_features * 2),
                nn.SiLU(),
                nn.BatchNorm1d(backbone_out_features * 2),

                # nn.Dropout(dropout / 2, inplace=True),
                nn.Linear(backbone_out_features * 2, num_classes),
            )

    def forward(self, x):
        features = self.backbone(x)

        if self.hyper:
            x = self.hyper(features)
        else:
            x = features[-1]

        if self.pre_attention:
            x = self.pre_attention(x)

        if self.dilated_conv:
            x = self.dilated_conv(x)

        if self.conv_head:
            x = self.conv_head(x)
            x = self.bn_head(x)
            x = self.silu(x)

        if self.attention:
            x = self.attention(x)

        if self.avg_pool:
            x = self.avg_pool(x)
            x = x.flatten(1)

        x = self.fc(x)

        return x
