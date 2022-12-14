# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding
###
import timm

# class ConvNeXt(nn.Module):
#     r""" ConvNeXt
#         A PyTorch impl of : `A ConvNet for the 2020s`  - https://arxiv.org/pdf/2201.03545.pdf

#     Args:
#         in_chans (int): Number of input image channels. Default: 3
#         num_classes (int): Number of classes for classification head. Default: 1000
#         depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
#         dims (tuple(int)): Feature dimension at each stage. Default: [96, 192, 384, 768]
#         drop_rate (float): Head dropout rate
#         drop_path_rate (float): Stochastic depth rate. Default: 0.
#         ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
#         head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
#     """

#     def __init__(
#             self,
#             in_chans=3,
#             num_classes=1000,
#             global_pool='avg',
#             output_stride=32,
#             depths=(3, 3, 9, 3),
#             dims=(96, 192, 384, 768),
#             kernel_sizes=7,
#             ls_init_value=1e-6,
#             stem_type='patch',
#             patch_size=4,
#             head_init_scale=1.,
#             head_norm_first=False,
#             conv_mlp=False,
#             conv_bias=True,
#             act_layer='gelu',
#             norm_layer=None,
#             drop_rate=0.,
#             drop_path_rate=0.,
#     ):
#         super().__init__()
#         assert output_stride in (8, 16, 32)
#         kernel_sizes = to_ntuple(4)(kernel_sizes)
#         if norm_layer is None:
#             norm_layer = LayerNorm2d
#             norm_layer_cl = norm_layer if conv_mlp else LayerNorm
#         else:
#             assert conv_mlp,\
#                 'If a norm_layer is specified, conv MLP must be used so all norm expect rank-4, channels-first input'
#             norm_layer_cl = norm_layer

#         self.num_classes = num_classes
#         self.drop_rate = drop_rate
#         self.feature_info = []

#         assert stem_type in ('patch', 'overlap', 'overlap_tiered')
#         if stem_type == 'patch':
#             # NOTE: this stem is a minimal form of ViT PatchEmbed, as used in SwinTransformer w/ patch_size = 4
#             self.stem = nn.Sequential(
#                 nn.Conv2d(in_chans, dims[0], kernel_size=patch_size, stride=patch_size, bias=conv_bias),
#                 norm_layer(dims[0])
#             )
#             stem_stride = patch_size
#         else:
#             mid_chs = make_divisible(dims[0] // 2) if 'tiered' in stem_type else dims[0]
#             self.stem = nn.Sequential(
#                 nn.Conv2d(in_chans, mid_chs, kernel_size=3, stride=2, padding=1, bias=conv_bias),
#                 nn.Conv2d(mid_chs, dims[0], kernel_size=3, stride=2, padding=1, bias=conv_bias),
#                 norm_layer(dims[0]),
#             )
#             stem_stride = 4

#         self.stages = nn.Sequential()
#         dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
#         stages = []
#         prev_chs = dims[0]
#         curr_stride = stem_stride
#         dilation = 1
#         # 4 feature resolution stages, each consisting of multiple residual blocks
#         for i in range(4):
#             stride = 2 if curr_stride == 2 or i > 0 else 1
#             if curr_stride >= output_stride and stride > 1:
#                 dilation *= stride
#                 stride = 1
#             curr_stride *= stride
#             first_dilation = 1 if dilation in (1, 2) else 2
#             out_chs = dims[i]
#             stages.append(ConvNeXtStage(
#                 prev_chs,
#                 out_chs,
#                 kernel_size=kernel_sizes[i],
#                 stride=stride,
#                 dilation=(first_dilation, dilation),
#                 depth=depths[i],
#                 drop_path_rates=dp_rates[i],
#                 ls_init_value=ls_init_value,
#                 conv_mlp=conv_mlp,
#                 conv_bias=conv_bias,
#                 act_layer=act_layer,
#                 norm_layer=norm_layer,
#                 norm_layer_cl=norm_layer_cl
#             ))
#             prev_chs = out_chs
#             # NOTE feature_info use currently assumes stage 0 == stride 1, rest are stride 2
#             self.feature_info += [dict(num_chs=prev_chs, reduction=curr_stride, module=f'stages.{i}')]
#         self.stages = nn.Sequential(*stages)
#         self.num_features = prev_chs

#         # if head_norm_first == true, norm -> global pool -> fc ordering, like most other nets
#         # otherwise pool -> norm -> fc, the default ConvNeXt ordering (pretrained FB weights)
#         self.norm_pre = norm_layer(self.num_features) if head_norm_first else nn.Identity()
#         # self.head = nn.Sequential(OrderedDict([
#         #         ('global_pool', SelectAdaptivePool2d(pool_type=global_pool)),
#         #         ('norm', nn.Identity() if head_norm_first else norm_layer(self.num_features)),
#         #         ('flatten', nn.Flatten(1) if global_pool else nn.Identity()),
#         #         ('drop', nn.Dropout(self.drop_rate)),
#         #         ('fc', nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())]))
#         self.head = nn.Sequential(OrderedDict([
#                 ('fc', nn.Linear(1024, 2048)),
#                 ('batch_norm2d', nn.BatchNorm2d(2048)),
#                 ('relu', nn.ReLU())
#                 ]))

#         named_apply(partial(_init_weights, head_init_scale=head_init_scale), self)

#     @torch.jit.ignore
#     def group_matcher(self, coarse=False):
#         return dict(
#             stem=r'^stem',
#             blocks=r'^stages\.(\d+)' if coarse else [
#                 (r'^stages\.(\d+)\.downsample', (0,)),  # blocks
#                 (r'^stages\.(\d+)\.blocks\.(\d+)', None),
#                 (r'^norm_pre', (99999,))
#             ]
#         )

#     @torch.jit.ignore
#     def set_grad_checkpointing(self, enable=True):
#         for s in self.stages:
#             s.grad_checkpointing = enable

#     @torch.jit.ignore
#     def get_classifier(self):
#         return self.head.fc

#     def reset_classifier(self, num_classes=0, global_pool=None):
#         if global_pool is not None:
#             self.head.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
#             self.head.flatten = nn.Flatten(1) if global_pool else nn.Identity()
#         self.head.fc = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

#     def forward_features(self, x):
#         x = self.stem(x)
#         x = self.stages(x)
#         x = self.norm_pre(x)
#         return x

#     def forward_head(self, x, pre_logits: bool = False):
#         # NOTE nn.Sequential in head broken down since can't call head[:-1](x) in torchscript :(
#         # x = self.head.global_pool(x)
#         # x = self.head.norm(x)
#         # x = self.head.flatten(x)
#         # x = self.head.drop(x)
#         x = self.head(x)
#         return x if pre_logits else self.head.fc(x)

#     def forward(self, x):
#         x = self.forward_features(x)
#         x = self.forward_head(x)
#         return x
###

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        # if return_interm_layers:
        #     return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        # else:
        #     return_layers = {'layer4': "0"}
        if return_interm_layers:
            return_layers = {"stages": "0", "stages": "1", "stages": "2", "stages": "3"}
        else:
            return_layers = {'stages': "3"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)
        
###
class Backbone_ConvNext(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        # backbone = getattr(torchvision.models, name)(
        #     replace_stride_with_dilation=[False, False, dilation],
        #     pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        # num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        # convnext.head = nn.Sequential(
        #     nn.Linear(1024, 2048),
        #     nn.BatchNorm2d(2048),
        #     nn.ReLU(),
        # )
        backbone = timm.create_model("convnext_base")
        num_channels = 1024
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)
###


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    # backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    backbone = Backbone_ConvNext(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
