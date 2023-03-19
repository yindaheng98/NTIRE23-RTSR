import torch

from .RFDNB4 import RFDNB4, RFDNB4_P
from .block import RFDBS, RFDBS_P, Cond2dSplit


class RFDNB4S(RFDNB4):
    def __init__(self, in_nc=3, nf=50, **kwargs):
        super().__init__(in_nc=in_nc, nf=nf, **kwargs)
        self.B2S1 = RFDBS(in_channels=nf)
        self.B2S2 = RFDBS(in_channels=nf)
        self.B2S3 = RFDBS(in_channels=nf)

    def forward(self, input):
        return self.tail(self.fea_conv(input), self.B2S1(self.fea_conv1(input)),
                         self.B2S2(self.fea_conv2(input)), self.B2S3(self.fea_conv3(input)))


class RFDNB4S_P(RFDNB4_P):
    def __init__(self, model: RFDNB4S, in_nc=3, nf=50):
        super().__init__(model, in_nc=in_nc, nf=nf)
        self.B2S123 = RFDBS_P([model.B2S1, model.B2S2, model.B2S3], in_channels=nf)
        self.split_left = Cond2dSplit(nf * 4, 0, nf)
        self.split_right = Cond2dSplit(nf * 4, nf, nf * 4)

    def forward(self, input):
        out_fea1234 = self.fea_conv1234(torch.cat([input, input, input, input], dim=1))
        out_fea1234 = torch.cat([
            self.split_left(out_fea1234),
            self.B2S123(self.split_right(out_fea1234))
        ], dim=1)
        return self.tail(out_fea1234)
