from typing import Tuple, Dict

import timm
import torch
import torch.nn.functional as F
import torchvision.models as models
from timm.models.features import FeatureInfo
from torch import nn, Tensor
from torch.cuda.amp import autocast

from model.aux_common import AuxCommon
from model.aux_skip_attention import AuxSkipAttention


def _timm(model_name):
    def _get_ef(pretrained, in_channels, num_classes, **kwargs):
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes,
                                  in_chans=in_channels)
        return model

    return _get_ef


class RainforestModel(nn.Module):
    def __init__(self, model_name: str, model_top:str, pretrained, use_coord_conv: bool, efficient_net_hyper_column,
                 num_in_channels, classes_num, backbone_name=None, input_height=0):
        super().__init__()

        self.classes_num = classes_num

        self.is_coord_model = model_name.startswith('coord_')
        # self.is_base_resnet_model = backbone_name.split('_')[0] not in ['coord', 'hyper', 'en']
        self.is_base_resnet_model = False
        self.is_efficient_net = model_name.startswith('en_')
        self.efficient_net_hyper_column = efficient_net_hyper_column
        self.is_aux = model_name in ('aux', 'aux2')

        backbone_classes = {
            # 'resnet50': models.resnet50,
            # 'resnet34': models.resnet34,
            # 'resnet18': models.resnet18,
            'resnet50': _timm('resnet50'),
            'resnet34': _timm('resnet34'),
            'resnet18': _timm('resnet18'),
            'en_b0_ns': _timm('tf_efficientnet_b0_ns'),
            'en_b1_ns': _timm('tf_efficientnet_b1_ns'),
            'en_b2_ns': _timm('tf_efficientnet_b2_ns'),
            'en_b3_ns': _timm('tf_efficientnet_b3_ns'),
            'en_b4_ns': _timm('tf_efficientnet_b4_ns'),
            'en_b5_ns': _timm('tf_efficientnet_b5_ns'),
            'en_b6_ns': _timm('tf_efficientnet_b6_ns'),
            'en_b7_ns': _timm('tf_efficientnet_b7_ns'),
            'x41': _timm('xception41'),
            'x71': _timm('xception71'),
            'resnest50d': _timm('resnest50d'),
            'seresnet152d_320': _timm('seresnet152d_320'),
            'aux': AuxSkipAttention,
            'aux2': AuxCommon,
        }
        extra_params_dict = {
            'hyper_resnet50': {'use_coord_conv': use_coord_conv},
            'hyper_resnet34': {'use_coord_conv': use_coord_conv},
        }

        backbone_class = backbone_classes[model_name]
        extra_param = extra_params_dict[model_name] if model_name in extra_params_dict else {}

        self.num_in_channels = num_in_channels

        if self.is_aux:
            if model_name == 'aux':
                self.backbone = AuxSkipAttention(self.classes_num, in_channels=self.num_in_channels)
            elif model_name == 'aux2':
                self.backbone = AuxCommon(backbone_name, model_top, self.classes_num, in_channels=self.num_in_channels,
                                          pretrained=pretrained, input_height=input_height)
        elif self.is_base_resnet_model:
            self.backbone = backbone_class(pretrained=pretrained)
            self.backbone.conv1 = nn.Conv2d(
                self.num_in_channels,
                self.backbone.conv1.out_channels,
                kernel_size=self.backbone.conv1.kernel_size,
                stride=self.backbone.conv1.stride,
                padding=self.backbone.conv1.padding,
                bias=False,
            )
        else:
            self.backbone = backbone_class(pretrained=pretrained, in_channels=self.num_in_channels, num_classes=0,
                                           **extra_param)

            if self.is_efficient_net and self.efficient_net_hyper_column:
                out_indices = (1, 2, 3, 4)  # (0, 1, 2, 3, 4)
                self.feature_info = FeatureInfo(self.backbone.feature_info, out_indices)
                self._stage_out_idx = {v['stage']: i for i, v in enumerate(self.feature_info) if i in out_indices}
                channels_info = [v['num_chs'] for i, v in enumerate(self.feature_info) if i in out_indices]
                in_channels = sum(channels_info)
                # self.hyper_conv = nn.Conv2d(in_channels, channels_info[-1], 1, padding=0)
                self.hyper_conv = nn.Conv2d(in_channels, channels_info[-1], 3, padding=1)
                self.hyper_bn = nn.BatchNorm2d(channels_info[-1])
                self.hyper_act = nn.SiLU(inplace=True)

        backbone_out_features = getattr(self.backbone, 'num_features', 0)  # EfficientNet from timm
        if backbone_out_features == 0 and self.is_aux is False:
            backbone_out_features = self.backbone.inplanes

        # You can add more layers here.

        if self.is_aux:
            self.head = nn.Identity()
            self.logit = nn.Identity()
        elif not model_top:
            self.head = nn.Identity()
            self.logit = nn.Linear(backbone_out_features, out_features=self.classes_num)
        elif model_top in ('v1', 'v2', 'v3'):
            mid_features = backbone_out_features // 2
            if model_top in ('v3',):
                mid_features = 1204

            self.head = nn.Sequential(
                nn.BatchNorm1d(num_features=backbone_out_features),
                nn.Dropout(0.4),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=backbone_out_features, out_features=mid_features),
                nn.BatchNorm1d(num_features=mid_features),
                nn.Dropout(0.4),
            )
            self.logit = nn.Linear(in_features=mid_features, out_features=self.classes_num, bias=True)
            if model_top in ('v1', 'v3'):
                self.logit.bias.data[:] = -2

        # self.logit = nn.Linear(backbone_out_features, out_features=self.classes_num)

    @autocast()
    def forward(self, x: Tensor):
        if self.is_base_resnet_model:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)

            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
        elif self.efficient_net_hyper_column and self.is_efficient_net:
            x = self.backbone.conv_stem(x)
            x = self.backbone.bn1(x)
            x = self.backbone.act1(x)

            # x = self.backbone.blocks(x)
            features = []
            if 0 in self._stage_out_idx:
                features.append(x)  # add stem out
            for i, b in enumerate(self.backbone.blocks):
                x = b(x)
                if i + 1 in self._stage_out_idx:
                    features.append(x)

            dst_size = features[0].size()[-2:]
            layers = []
            for f in features:
                layers.append(F.interpolate(f, size=dst_size, mode='bilinear', align_corners=False))
            x = torch.cat(layers, dim=1)

            x = self.hyper_conv(x)
            x = self.hyper_bn(x)
            x = self.hyper_act(x)

            x = self.backbone.conv_head(x)
            x = self.backbone.bn2(x)
            x = self.backbone.act2(x)
            x = self.backbone.global_pool(x)
            x = self.backbone.classifier(x)
        else:
            x = self.backbone(x)

        x = self.head(x)
        x = self.logit(x)

        return x
