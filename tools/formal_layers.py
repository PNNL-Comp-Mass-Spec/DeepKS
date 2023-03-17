from torch.nn import Module
from torch import concat, squeeze, unsqueeze, mul, transpose


class Concatenation(Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, *inputs):
        return concat(inputs, dim=self.dim)


class Squeeze(Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return squeeze(x, dim=self.dim)


class Unsqueeze(Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return unsqueeze(x, dim=self.dim)


class Multiply(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return mul(x, y)


class Transpose(Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return transpose(x, self.dim1, self.dim2)
