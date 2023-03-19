import torch
from models.rfdn_half.RFDNB2S import RFDNB2S_P
from models.rfdn_half.RFDNB4S import RFDNB4S_P


def srmodel():
    return torch.load("PRFDN_x2.pt.pth")
