import torch

from .RFDNB2 import RFDNB2, RFDNB2_P
from .block import RFDBS, Cond2dSplit


class RFDNB2S(RFDNB2):
    def __init__(self, in_nc=3, nf=50, **kwargs):
        super().__init__(in_nc=in_nc, nf=nf, **kwargs)
        self.B2S2 = RFDBS(in_channels=nf)

    def forward(self, input):
        return self.tail(self.fea_conv(input), self.B2S2(self.fea_conv2(input)))


class RFDNB2S_P(RFDNB2_P):
    def __init__(self, model: RFDNB2S, in_nc=3, nf=50):
        super().__init__(model, in_nc=in_nc, nf=nf)
        self.B2S2 = model.B2S2
        self.split_left = Cond2dSplit(nf * 2, 0, nf)
        self.split_right = Cond2dSplit(nf * 2, nf, nf * 2)

    def forward(self, input):
        out_fea12 = self.fea_conv12(torch.cat([input, input], dim=1))
        out_fea12 = torch.cat([
            self.split_left(out_fea12),
            self.B2S2(self.split_right(out_fea12))
        ], dim=1)
        return self.tail(out_fea12)
