import torch
import torch.nn as nn

import models.rfdn_baseline.block as B
from .RFDNB2 import RFDNB2
from .block import RFDB_P, conv_layer_p, Cond2dSplit


class RFDNB4(RFDNB2):
    def __init__(self, in_nc=3, nf=50, **kwargs):
        super().__init__(in_nc=in_nc, nf=nf, **kwargs)
        self.fea_conv1 = B.conv_layer(in_nc, nf, kernel_size=3)
        self.fea_conv3 = B.conv_layer(in_nc, nf, kernel_size=3)

    def forward(self, input):
        return self.tail(self.fea_conv(input), self.fea_conv1(input), self.fea_conv2(input), self.fea_conv3(input))

    def tail(self, out_fea, out_fea1, out_fea2, out_fea3):
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_fea1)
        out_B3 = self.B3(out_fea2)
        out_B4 = self.B4(out_fea3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output


class RFDNB4_P(nn.Module):
    def __init__(self, model: RFDNB4, in_nc=3, nf=50):
        super().__init__()

        self.fea_conv1234 = conv_layer_p([model.fea_conv, model.fea_conv1, model.fea_conv2, model.fea_conv3],
                                         in_nc, nf, kernel_size=3)
        self.split = Cond2dSplit(nf * 4, 0, nf)

        self.B1234 = RFDB_P([model.B1, model.B2, model.B3, model.B4], in_channels=nf)
        self.c = model.c
        self.LR_conv = model.LR_conv
        self.upsampler = model.upsampler
        self.scale_idx = 0

    def forward(self, input):
        return self.tail(self.fea_conv1234(torch.cat([input, input, input, input], dim=1)))

    def tail(self, out_fea1234):
        out_B1234 = self.B1234(out_fea1234)

        out_B = self.c(out_B1234)
        out_lr = self.LR_conv(out_B) + self.split(out_fea1234)

        output = self.upsampler(out_lr)

        return output
