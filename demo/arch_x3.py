import torch

"""
!!!Important!!!
The structure of our model is in demo/utils/models.
Since applied auto-pruning our model, models in .pth file have same structure of the models in demo/utils/models,
but have different shape of weights.
So our model can only be loaded by `torch.load` rather than `model.load_state_dict`
"""


def srmodel():
    return torch.load("PRFDN_x3.pth")
