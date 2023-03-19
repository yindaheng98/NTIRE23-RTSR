import torch
import torch.nn as nn

import models.rfdn_baseline.block as B
from models.rfdn_baseline.block import conv_layer, activation, ESA,get_valid_padding,pad,norm,sequential
import math

class RFDBS(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(RFDBS, self).__init__()
        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.c4 = conv_layer(self.remaining_channels, self.dc, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.dc * 2, in_channels, 1)
        self.esa = ESA(in_channels, nn.Conv2d)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1 + input)

        r_c4 = self.act(self.c4(r_c1))

        out = torch.cat([distilled_c1, r_c4], dim=1)
        out_fused = self.esa(self.c5(out))

        return out_fused


set0 = False


def conv_p(convs: list[nn.Conv2d], in_channels, out_channels, kernel_size, stride=1, padding=0, **kwargs):
    conv = nn.Conv2d(in_channels * len(convs), out_channels * len(convs),
                     kernel_size, stride=stride, padding=padding, **kwargs)
    with torch.no_grad():
        for i in range(len(convs)):
            for j in range(len(convs)):
                if j == i:
                    conv.weight[j * out_channels:(j + 1) * out_channels, i * in_channels:(i + 1) * in_channels, ...] = \
                        convs[i].weight
                elif set0:
                    conv.weight[j * out_channels:(j + 1) * out_channels, i * in_channels:(i + 1) * in_channels, ...] = \
                        torch.zeros(convs[i].weight.shape)
        for i in range(len(convs)):
            conv.bias[i * out_channels:(i + 1) * out_channels] = convs[i].bias
    return conv


def conv_layer_p(convs: list[nn.Conv2d], in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return conv_p(convs, in_channels, out_channels,
                  kernel_size, stride, padding=padding,
                  bias=True, dilation=dilation, groups=groups)

def conv_layer_expand(conv,in_channels,out_channels,expand,kernel_size,flag=True,**kwargs):
    in_exp = math.ceil(in_channels*expand[0])
    out_exp= math.ceil(out_channels*expand[1])
    if flag:
        conv_exp = conv_layer(in_exp,out_exp,kernel_size=kernel_size,**kwargs)
    else:
        conv_exp=nn.Conv2d(in_channels=in_exp,out_channels=out_exp,kernel_size=kernel_size,**kwargs)
    with torch.no_grad():
        conv_exp.weight[:,:,...]=torch.zeros(conv_exp.weight.shape)
        conv_exp.weight[0:out_channels,0:in_channels,...]=conv.weight
        conv_exp.bias[:]=torch.zeros(conv_exp.bias.shape)
        conv_exp.bias[0:out_channels]=conv.bias
    return conv_exp

# def conv_block_exp(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
#                pad_type='zero', norm_type=None, act_type='relu'):
#     padding = get_valid_padding(kernel_size, dilation)
#     p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
#     padding = padding if pad_type == 'zero' else 0

#     c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
#                   dilation=dilation, bias=bias, groups=groups)
#     a = activation(act_type) if act_type else None
#     n = norm(norm_type, out_nc) if norm_type else None
#     return sequential(p, c, n, a)


def conv_block_exp(in_nc, out_nc, kernel_size=1, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu',model=None):
    
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = conv_layer_expand(model[0],in_nc,out_nc,[2,2],kernel_size=kernel_size,stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups,flag=False)
    
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)

# upsample_block(nf, out_nc, upscale_factor=upscale,model=model.upsampler)

def pixelshuffle_block_exp(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1,model=None):
    conv = conv_layer_expand(model[0],in_channels,out_channels* (upscale_factor ** 2),[2,1], kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)

class ESA_EXP(ESA):
    def __init__(self, n_feats,model:None):
        super().__init__(n_feats,nn.Conv2d)
        f = n_feats // 4
        self.conv1 = conv_layer_expand(model.conv1,n_feats, f,[2,2],kernel_size=1,flag=False)
        self.conv_f = conv_layer_expand(model.conv_f, f, f,[2,2],kernel_size=1,flag=False)
        self.conv_max = conv_layer_expand(model.conv_max, f, f,[2,2],kernel_size=3, padding=1,flag=False)
        self.conv2 = conv_layer_expand(model.conv2,f, f,[2,2],kernel_size=3, stride=2, padding=0,flag=False)
        self.conv3 = conv_layer_expand(model.conv3, f, f,[2,2], kernel_size=3, padding=1,flag=False)
        self.conv3_ = conv_layer_expand(model.conv3_, f, f,[2,2] ,kernel_size=3, padding=1,flag=False)
        self.conv4 = conv_layer_expand(model.conv4, f, n_feats,[2,2], kernel_size=1,flag=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)


class ESA_P(ESA):
    def __init__(self, models: list[ESA], n_feats, conv):
        super().__init__(n_feats * len(models), nn.Conv2d)
        f = n_feats // 4
        self.conv1 = conv([m.conv1 for m in models], n_feats, f, kernel_size=1)
        self.conv_f = conv([m.conv_f for m in models], f, f, kernel_size=1)
        self.conv_max = conv([m.conv_max for m in models], f, f, kernel_size=3, padding=1)
        self.conv2 = conv([m.conv2 for m in models], f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv([m.conv3 for m in models], f, f, kernel_size=3, padding=1)
        self.conv3_ = conv([m.conv3_ for m in models], f, f, kernel_size=3, padding=1)
        self.conv4 = conv([m.conv4 for m in models], f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)


class Conv2dCat(nn.Conv2d):
    def __init__(self, n: int, in_channels: list[int]):
        for in_channel in in_channels:
            assert in_channel % n == 0
        super().__init__(in_channels=sum(in_channels), out_channels=sum(in_channels), kernel_size=1)
        with torch.no_grad():
            self.weight[...] = torch.zeros(self.weight.shape)
            self.bias[...] = torch.zeros(self.bias.shape)
            for tensor_i in range(len(in_channels)):  # 哪个tensor
                channels_of_tensor = in_channels[tensor_i]  # 这个tensor有多少channel
                channels_to_move = channels_of_tensor // n  # 以多少channel为单位进行移动
                channel_start_i_of_tensor = sum(in_channels[0:tensor_i])  # 这个tensor的channel开始于何处
                for split_i in range(n):  # 第几批移动
                    # 这一批要移动哪些channel
                    src_idx = channel_start_i_of_tensor + split_i * channels_to_move
                    # 这一批channel要移动到哪
                    dst_idx = split_i * sum(in_channels) // n + sum(in_channels[0:tensor_i]) // n
                    self.weight[dst_idx:dst_idx + channels_to_move, src_idx:src_idx + +channels_to_move, 0, 0] = \
                        torch.eye(channels_to_move)

    def forward(self, *inputs):
        return super().forward(torch.cat(inputs, dim=1))


class RFDB_P(B.RFDB):
    def __init__(self, models: list[B.RFDB], in_channels, distillation_rate=0.25):
        super(RFDB_P, self).__init__(in_channels * len(models), distillation_rate=distillation_rate)
        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer_p([m.c1_d for m in models], in_channels, self.dc, 1)
        self.c1_r = conv_layer_p([m.c1_r for m in models], in_channels, self.rc, 3)
        self.c2_d = conv_layer_p([m.c2_d for m in models], self.remaining_channels, self.dc, 1)
        self.c2_r = conv_layer_p([m.c2_r for m in models], self.remaining_channels, self.rc, 3)
        self.c3_d = conv_layer_p([m.c3_d for m in models], self.remaining_channels, self.dc, 1)
        self.c3_r = conv_layer_p([m.c3_r for m in models], self.remaining_channels, self.rc, 3)
        self.c4 = conv_layer_p([m.c4 for m in models], self.remaining_channels, self.dc, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer_p([m.c5 for m in models], self.dc * 4, in_channels, 1)
        self.esa = ESA_P([m.esa for m in models], in_channels, conv_p)

        self.cat_p = Conv2dCat(n=len(models), in_channels=[self.dc * len(models)] * 4)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1 + input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2 + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3 + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = self.cat_p(distilled_c1, distilled_c2, distilled_c3, r_c4)
        out_fused = self.esa(self.c5(out))

        return out_fused


class RFDBS_P(nn.Module):
    def __init__(self, models: list[RFDBS], in_channels, distillation_rate=0.25):
        super(RFDBS_P, self).__init__()
        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer_p([m.c1_d for m in models], in_channels, self.dc, 1)
        self.c1_r = conv_layer_p([m.c1_r for m in models], in_channels, self.rc, 3)
        self.c4 = conv_layer_p([m.c4 for m in models], self.remaining_channels, self.dc, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer_p([m.c5 for m in models], self.dc * 2, in_channels, 1)
        self.esa = ESA_P([m.esa for m in models], in_channels, conv_p)

        self.cat_p = Conv2dCat(n=len(models), in_channels=[self.dc * len(models)] * 2)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1 + input)

        r_c4 = self.act(self.c4(r_c1))

        out = self.cat_p(distilled_c1, r_c4)
        out_fused = self.esa(self.c5(out))

        return out_fused


class Cond2dSplit(nn.Conv2d):
    def __init__(self, in_channels, i, j):
        super().__init__(in_channels=in_channels, out_channels=j - i, kernel_size=1)
        assert i < j <= in_channels
        with torch.no_grad():
            self.weight[...] = torch.zeros(self.weight.shape)
            self.bias[...] = torch.zeros(self.bias.shape)
            self.weight[0:j - i, i:j, 0, 0] = torch.eye(j - i)

    def forward(self, input):
        return super().forward(input)
