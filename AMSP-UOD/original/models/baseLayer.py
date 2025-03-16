import torch 
import torch.nn as nn
import numpy as np
import math


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

class BrokenBlock(nn.Module):
    """ 比起Tensorflow2.x 没想到还是Torch更反人类 """
    def __init__(self,  dim_len, dim=1, dim_shape=4, group=1, *args, **kwargs):
        super(BrokenBlock, self).__init__(*args, **kwargs)
        self.dim = dim
        self.group = group
        self.dim_len = dim_len
        self.view = list(np.ones(dim_shape, int))

    def getShuffle(self, shape):
        # 生成随机索引
        self.view[self.dim] = shape[self.dim]
        perm = torch.randperm(self.dim_len//self.group).unsqueeze(1)
        indices = torch.cat([perm * self.group + _ for _ in range(self.group)], dim=1)
        indices = indices.view(self.view).expand(shape)
        return indices

    def forward(self, x):
        if self.training:
            return torch.gather(x, self.dim, self.getShuffle(x.shape).to(x.device))
        else:
            return x
        
class SpiralConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g=4, act=act)
        self.brk = BrokenBlock(c_, group=c_//4)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)
        self.cv3 = Conv(2 * c_ ,c2, 1, 1, act=act)

    def forward(self, x):
        x = self.cv1(x)
        y = self.brk(x)
        return self.cv3(torch.cat((x, self.cv2(y)), 1))


class DW(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, *args, **kwargs):
        super(DW, self).__init__(*args, **kwargs)
        self.cv1 = nn.Conv2d(c1, c1, k, s, autopad(k, p, d), groups=c1)
        self.cv2 = nn.Conv2d(c1, c2, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        y = self.act(self.cv1(x))
        y = self.act(self.bn(self.cv2(y)))
        return y

class SCL(nn.Module):
    def __init__(self, in_channels, out_channels, k=1, s=1, p=None, g=1, d=1, act=True, length=4, *args, **kwargs):
        super(SCL, self).__init__(*args, **kwargs)
        self.length = length
        self.c = in_channels//length
        self.cv1 = nn.Conv2d(self.c, self.c, 3, 1, 1)
        self.cv2 = nn.Conv2d(self.c, self.c, k, s, autopad(k,p,d),d)
        self.cv3 = nn.Conv2d(in_channels, out_channels, bias=False)

        self.brk = BrokenBlock(in_channels, group=self.c)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        if self.training:
            y = x.chunk(self.length, 1)
            y = [self.act(self.cv1(_)) for _ in y]
            x = self.brk(torch.cat(y, 1))
            z = x.chunk(self.length, 1)
            z = [self.act(self.cv2(_)) for _ in z]
            z = torch.cat(z, 1)
        else:
            x = x.chunk(self.length, 1)
            y = [self.act(self.cv1(_)) for _ in x]
            z = [self.act(self.cv2(_)) for _ in y]
            z = torch.cat(z, 1)
        out = self.act(self.bn(self.cv3(z)))
        return out


class SC(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, length=4, *args, **kwargs):
        super(SC, self).__init__(*args, **kwargs)
        self.length = length
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = nn.Conv2d(c_//length, c_//length, 5, 1, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c_)
        self.act = nn.SiLU()
        self.brk = BrokenBlock(c_, group=c_//length)

    def forward(self, x):
        x = self.cv1(x)
        if self.training:
            y = self.brk(x)
        else:
            y = x
        y = y.chunk(self.length, 1)
        out = [self.cv2(_) for _ in y]
        out = self.act(self.bn(torch.cat(out, 1)))
        return torch.cat([x, out], 1)
    

class BKSC(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, length=4, *args, **kwargs):
        super(SC, self).__init__(*args, **kwargs)
        self.length = length
        self.cv1 = Conv(c1, c2, k, s, None, length, act=act)
        self.cv2 = nn.Conv2d(c2//length, c2//length, 5, 1, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.brk = BrokenBlock(c2, group=c2//length)

    def forward(self, x):
        x = self.cv1(x)
        if self.training:
            y = self.brk(x)
        else:
            y = x
        y = y.chunk(self.length, 1)
        out = [self.cv2(_) for _ in y]
        out = self.act(self.bn(torch.cat(out, 1)))
        return out + x

class Spiral(nn.Module):
    def __init__(self, in_channels, out_channels, k=1, s=1, p=None, g=1, d=1, act=True, length=4, *args, **kwargs):
        super(Spiral, self).__init__(*args, **kwargs)
        self.length = length
        self.act = nn.SiLU()

        self.cv1 = nn.Conv2d(in_channels, in_channels, k, s, autopad(k, p, d), d, groups=length, bias=False)
        self.brk = BrokenBlock(dim_len=in_channels, group=in_channels//length)
        self.cv2 = nn.Conv2d(in_channels, in_channels, 5, 1 ,2, 1, groups=length, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)

        self.cv3 = nn.Conv2d(in_channels, out_channels, 1, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = self.act(self.cv1(x))
        if self.training:
            y = self.brk(y)
        y = self.cv2(y)
        y = self.bn2(y)
        y = self.act(y)
        if x.shape == y.shape:
            out = self.act(self.bn3(self.cv3(y + x)))
        else:
            out = self.act(self.bn3(self.cv3(y)))
        return out


class PoolBlock(nn.Module):
    def __init__(self, size):
        super(PoolBlock, self).__init__()
        self.avg = nn.AdaptiveAvgPool2d(size)
        self.max = nn.AdaptiveMaxPool2d(size)

    def forward(self, x):
        avg = self.avg(x)
        max = self.max(x)
        return avg + max


class PWConv(nn.Module):
    def __init__(self, in_channels, out_channels, ratio = 32):
        super(PWConv, self).__init__()
        self.csp = Conv(in_channels, out_channels)
        self.pool_h, self.pool_w = PoolBlock((1, None)), PoolBlock((None, 1))
        c = max(8, in_channels//ratio)
        self.cv1 = nn.Conv2d(in_channels=out_channels, out_channels=c, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(c)
        self.act1 = nn.Hardswish()

        self.cv2 = nn.Conv2d(in_channels=c, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.cv3 = nn.Conv2d(in_channels=c, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.act2 = nn.Sigmoid()

        self.brk = BrokenBlock(out_channels, group=2)

    def forward(self, x):
        x = self.csp(x)
        b,c,h,w = x.shape
        x_h, x_w = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2)

        x_tmp = torch.cat([x_h, x_w], dim=-1)
        x_bk = self.brk(x_tmp)
        x_bk = self.act1(self.bn(self.cv1(x_bk)))

        out_h, out_w = torch.split(x_bk, [h,w], dim=-1)
        out_h = self.cv2(out_h)
        out_w = self.cv3(out_w)
        out = out_w * out_h.permute(0, 1, 3, 2)

        return x * self.act2(out)



if __name__ == '__main__':
    matrix = torch.tensor([[[1,2,3],[4,5,6],[7,8,9]]])
    print(matrix)
    test = BrokenBlock(matrix.shape, 2)
    matrix = test(matrix)
    print(matrix)

