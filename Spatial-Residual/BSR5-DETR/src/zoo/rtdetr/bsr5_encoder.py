import copy
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from .utils import get_activation

from src.core import register
from src.nn.backbone.bsr5 import SRConv, SRViT, BSR5DETR_cfg

__all__ = ['BSR5Encoder']

'''
  - [-1, 1, SRViT, [256, 1024]] # 5

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 3], 1, Concat, [1]]  # cat backbone P4
  - [-1, 3, C2f, [512]]  # 8

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 2], 1, Concat, [1]]  # cat backbone P3
  - [-1, 3, C2f, [256]]  # 11 (P3/8-small)

  - [-1, 1, SRConv, [256, 3, 2]]
  - [[-1, 8], 1, Concat, [1]]  # cat head P4
  - [-1, 3, C2f, [512]]  # 14 (P4/16-medium)

  - [-1, 1, SRConv, [512, 3, 2]]
  - [[-1, 5], 1, Concat, [1]]  # cat head P5
  - [-1, 3, C2f, [1024]]  # 17 (P5/32-large)

  - [[11, 14, 17], 1, RTDETRDecoder, [nc]]  # Detect(P3, P4, P5)
'''


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            groups=g,
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=3, e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvNormLayer(c1, c_, k[0], 1, act=nn.SiLU())
        self.cv2 = ConvNormLayer(c_, c2, k[1], 1, g=g, act=nn.SiLU())
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = ConvNormLayer(c1, 2 * self.c, 1, 1, act=nn.SiLU())
        self.cv2 = ConvNormLayer((2 + n) * self.c, c2, 1, 1, act=nn.SiLU())  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    

@register
class BSR5Encoder(nn.Module):
    def __init__(self,
                in_channels=[512, 1024, 512],  # P3, P4, P5 from backbone
                #  feat_strides=[8, 16, 32],
                out_channels=[256, 512, 1024],
                # SRViT params
                max_dim=1024,
                hidden_dim=256,
                act='silu',
                info_print=False,
                ):
        super().__init__()
        self.in_channels = in_channels
        # self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        
        # Output channels: [P3, P4, P5]
        self.out_channels = out_channels
        # self.out_strides = feat_strides
        
        # SRViT layer for processing the last feature map
        self.srvit = SRViT(in_channels[-1], self.hidden_dim, max_dim, act=get_activation(act)) # 5: 512 -> 256
        print(f'number of trainable params in SRViT: {sum(p.numel() for p in self.srvit.parameters() if p.requires_grad)}')
        # Top-down FPN modules
        self.fpn_blocks = nn.ModuleList()
        self.fpn_blocks.append(
            C2f(in_channels[-2] + self.hidden_dim, self.out_channels[-2], n=3) # 8: 1024 + 256 -> 512
        )
        self.fpn_blocks.append(
            C2f(in_channels[-3] + self.out_channels[-2], self.out_channels[-3], n=3) # 11: 512 + 512 -> 256
        )
        if info_print:
            print(f'number of trainable params in FPN block 0: {sum(p.numel() for p in self.fpn_blocks[0].parameters() if p.requires_grad)}')
            print(f'number of trainable params in FPN block 1: {sum(p.numel() for p in self.fpn_blocks[1].parameters() if p.requires_grad)}')
        # Bottom-up PAN modules
        self.downsample_convs = nn.ModuleList()
        self.downsample_convs.append(
                SRConv(self.out_channels[-3], self.out_channels[-3], k=3, s=2, act=get_activation(act)) # 12: 256 -> 256
        )
        self.downsample_convs.append(
                SRConv(self.out_channels[-2], self.out_channels[-2], k=3, s=2, act=get_activation(act)) # 15: 512 -> 512
        )

        if info_print:
            print(f'number of trainable params in downsample block 0: {sum(p.numel() for p in self.downsample_convs[0].parameters() if p.requires_grad)}')
            print(f'number of trainable params in downsample block 1: {sum(p.numel() for p in self.downsample_convs[1].parameters() if p.requires_grad)}')

        self.pan_blocks = nn.ModuleList()
        self.pan_blocks.append(
                C2f(self.out_channels[-3] + self.out_channels[-2], self.out_channels[-2], n=3) # 14: 512 + 256 -> 512
        )
        self.pan_blocks.append(
                C2f(self.out_channels[-2] + self.hidden_dim, self.out_channels[-1], n=3) # 17: 256 + 256 -> 1024
        )
        if info_print:
            print(f'number of trainable params in pan block 0: {sum(p.numel() for p in self.pan_blocks[0].parameters() if p.requires_grad)}')
            print(f'number of trainable params in pan block 1: {sum(p.numel() for p in self.pan_blocks[1].parameters() if p.requires_grad)}')


    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        
        # # Apply input projection to match hidden dimension
        # proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        
        # Process the last feature map with SRViT
        srvit_out = self.srvit(feats[-1])
        # Top-down FPN path
        fpn_outs = [srvit_out]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = fpn_outs[-1]
            feat_low = feats[idx - 1]
            # Upsample high-level features
            upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
            
            # Concatenate with low-level features and process with FPN block
            fpn_out = self.fpn_blocks[len(self.in_channels)-1-idx](
                torch.concat([upsample_feat, feat_low], dim=1)
            )
            fpn_outs.append(fpn_out)

        # Bottom-up PAN path
        outs = [fpn_outs[-1]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            # Downsample low-level features
            downsample_feat = self.downsample_convs[idx](feat_low)
            concat_feat = torch.concat([downsample_feat, fpn_outs[len(self.in_channels) - 2 - idx]], dim=1)
            out = self.pan_blocks[idx](concat_feat)
            outs.append(out)

        return outs

