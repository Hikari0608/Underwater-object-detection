'''by lyuwenyu
'''
import torch
import torch.nn as nn 
import torch.nn.functional as F 

from collections import OrderedDict

from .common import get_activation, ConvNormLayer, FrozenBatchNorm2d

from src.core import register


__all__ = ['PResNet']


'''
backbone:
  # [from, repeats, module, args]
  - [-1, 1, LCPP, [64, 3, 2, 1]]  # 1-P2/4
  - [-1, 1, LCPP, [256, 3, 2, 1]]  # 1-P2/4
  - [-1, 1, LCPP, [512, 3, 2, 1]]  # 2-P3/8
  - [-1, 1, LCPP, [1024, 3, 2, 1]]  # 3-P4/16
  - [-1, 1, LCPP, [512, 3, 2, 1]]  # 4-P5/32
'''

BSR5DETR_cfg = {
    512: [64, 256, 512, 512, 512],
    1024: [64, 256, 512, 1024, 512],
}

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class SRConv(nn.Module):
    def __init__(self, ch_in, ch_out, k=3, s=1, e=0.5, split_rate=4, kk=5, act=nn.SiLU()):
        super().__init__()
        c_ = int(e * ch_out)
        self.ch_mid = c_ // split_rate

        self.cv1 = Conv(ch_in, c_, k, s, act=act)
        self.cv2 = Conv(self.ch_mid * 2, self.ch_mid * 2, 3, 1, act=act)
        self.cv3 = Conv(self.ch_mid * 1, self.ch_mid * 1, 3, 1, act=act)
        self.cv4 = Conv(c_ + self.ch_mid, ch_out, 1, 1, act=act)

        self.m = nn.MaxPool2d(kernel_size=kk, stride=1, padding=kk // 2)

    def forward(self, x):
        
        y = self.cv1(x)
        y1 = self.cv2(y[:,:self.ch_mid * 2])
        y2 = self.cv3(y1[:,:self.ch_mid * 1])

        return self.cv4(torch.cat([self.m(y2), y2, y1[:,self.ch_mid * 1:self.ch_mid * 2], y[:, self.ch_mid * 2:]], 1))


class TransformerEncoderLayer(nn.Module):
    """Defines a single layer of the transformer encoder."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        """Initialize the TransformerEncoderLayer with specified parameters."""
        super().__init__()
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor, pos=None):
        """Add position embeddings to the tensor if provided."""
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with post-normalization."""
        q = k = self.with_pos_embed(src, pos)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with pre-normalization."""
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        return src + self.dropout2(src2)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Forward propagates the input through the encoder module."""
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class AIFI(TransformerEncoderLayer):
    """Defines the AIFI transformer layer."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        """Initialize the AIFI instance with specified parameters."""
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)

    def forward(self, x):
        """Forward pass for the AIFI transformer layer."""
        c, h, w = x.shape[1:]
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c)
        # Flatten [B, C, H, W] to [B, HxW, C]
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed.to(device=x.device, dtype=x.dtype))
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        """Builds 2D sine-cosine position embedding."""
        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]


class TransformerLayer(nn.Module):
    """Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)."""

    def __init__(self, c, num_heads):
        """Initializes a self-attention mechanism using linear transformations and multi-head attention."""
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        """Apply a transformer block to the input x and return the output."""
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        return self.fc2(self.fc1(x)) + x


class SRViT(nn.Module):
    def __init__(self, ch_in, ch_out, ch_mid, k=3, s=1, e=1, split_rate=2, act=nn.SiLU()):
        super().__init__()
        c_ = int(e * ch_out)
        self.c = c_ // split_rate
        self.cv1 = Conv(ch_in, c_, k, s, 1, 1, act=act)
        self.cv2 = Conv(c_ + self.c, ch_out, 1, 1, act=act)

        self.m = AIFI(self.c * 1, ch_mid, 8)

    def forward(self, x):
        y = self.cv1(x)
        return self.cv2(torch.cat([self.m(y[:,:self.c * 1]), y], 1))
    

# class SpatialResResidualBlocks(nn.Module):
#     def __init__(self, block, ch_in, ch_out, count, stage_num, act='relu', variant='b'):
#         super().__init__()

#         self.blocks = nn.ModuleList()
#         for i in range(count):
#             self.blocks.append(
#                 block(
#                     ch_in, 
#                     ch_out,
#                     stride=2 if i == 0 and stage_num != 2 else 1, 
#                     shortcut=False if i == 0 else True,
#                     variant=variant,
#                     act=act)
#             )

#             if i == 0:
#                 ch_in = ch_out * block.expansion

#     def forward(self, x):
#         out = x
#         for block in self.blocks:
#             out = block(out)
#         return out


@register
class BackbonewithSpatialResidual(nn.Module):
    def __init__(
        self, 
        max_dim=512, 
        num_stages=5, 
        return_idx=[1, 2, 3, 4], 
        act='silu',
        freeze_at=-1, 
        freeze_norm=True, 
        pretrained=False):
        super().__init__()

        ch_in = 3
        ch_out_list = BSR5DETR_cfg[max_dim]

        self.stages = nn.ModuleList()
        for i in range(num_stages):
            self.stages.append(
                SRConv(ch_in, ch_out_list[i], 3, 2, 1, act=act)
            )
            print(f'stage {i} with ch_in {ch_in} and ch_out {ch_out_list[i]}')
            ch_in = ch_out_list[i]

        self.return_idx = return_idx
        self.out_channels = [ch_out_list[_i] for _i in return_idx]
        # self.out_strides = [_out_strides[_i] for _i in return_idx]

        if freeze_at >= 0:
            for i in range(min(freeze_at, num_stages)):
                self._freeze_parameters(self.stages[i])

        if freeze_norm:
            self._freeze_norm(self)

        if True:# info_print:
            for idx, _ in enumerate(self.stages):
                n_parameters = sum(p.numel() for p in _.parameters() if p.requires_grad)
                print(f'number of trainable params in stage {idx}: {n_parameters}')

            n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print('number of trainable params in backbone:', n_parameters)

            
    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m

    def forward(self, x):
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs


