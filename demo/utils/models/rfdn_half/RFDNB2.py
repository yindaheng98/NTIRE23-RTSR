import torch
import torch.nn as nn

import models.rfdn_baseline.block as B
from .RFDN import RFDN
from .block import RFDB_P, conv_layer_p, Conv2dCat, Cond2dSplit


class RFDNB2(RFDN):
    def __init__(self, in_nc=3, nf=50, **kwargs):
        super().__init__(in_nc=in_nc, nf=nf, **kwargs)
        self.fea_conv2 = B.conv_layer(in_nc, nf, kernel_size=3)

    def forward(self, input):
        return self.tail(self.fea_conv(input), self.fea_conv2(input))

    def tail(self, out_fea, out_fea2):
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_fea2)
        out_B4 = self.B4(out_B3)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output


class RFDNB2_P(nn.Module):
    def __init__(self, model: RFDNB2, in_nc=3, nf=50):
        super().__init__()

        self.fea_conv12 = conv_layer_p([model.fea_conv, model.fea_conv2],
                                       in_nc, nf, kernel_size=3)
        self.split = Cond2dSplit(nf * 2, 0, nf)

        self.B13 = RFDB_P([model.B1, model.B3], in_channels=nf)
        self.B24 = RFDB_P([model.B2, model.B4], in_channels=nf)
        self.cat_p = Conv2dCat(n=2, in_channels=[nf * 2] * 2)
        self.c = model.c
        self.LR_conv = model.LR_conv
        self.upsampler = model.upsampler
        self.scale_idx = 0

    def forward(self, input):
        return self.tail(self.fea_conv12(torch.cat([input, input], dim=1)))

    def tail(self, out_fea12):
        out_B13 = self.B13(out_fea12)
        out_B24 = self.B24(out_B13)

        out_B = self.c(self.cat_p(out_B13, out_B24))
        out_lr = self.LR_conv(out_B) + self.split(out_fea12)

        output = self.upsampler(out_lr)

        return output
