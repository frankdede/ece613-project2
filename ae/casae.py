import torch.nn as nn


class CASAE(nn.Module):
    def __init__(self, ae1, ae2):
        super(CASAE, self).__init__()
        self.ae1 = ae1
        self.ae2 = ae2

    def forward(self, input):
        y = input
        y = self.ae1(y).reshape(1, 1, 512, 512)
        y = self.ae2(y)
        return y
