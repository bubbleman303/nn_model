import numpy as np
import src.models.functions as fs


class MullLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class SigmoidLayer:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, d_out):
        return d_out * self.out * (1 - self.out)


class ReluLayer:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, d_out):
        d_out[self.mask] = 0
        dx = d_out
        return dx


class AffineLayer:
    def __init__(self, w, b):
        self.w = w
        self.b = b
        self.x = None
        self.dw = None
        self.db = None

    def forward(self, x):
        x_copy = x.copy()
        if x_copy.ndim == 1:
            x_copy = x_copy.reshape((1, x_copy.size))
        self.x = x_copy
        out = np.dot(x_copy, self.w) + self.b

        return out

    def backward(self, d_out):
        dx = np.dot(d_out, self.w.T)
        self.dw = np.dot(self.x.T, d_out)
        self.db = np.sum(d_out, axis=0)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = fs.softmax(x)
        self.loss = fs.cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, d_out=1):
        batch_size = self.t.shape[0]
        return (self.y - self.t) / batch_size * d_out


class MeanSquareLoss:
    def __init__(self):
        self.x = None
        self.t = None
        self.loss = None

    def forward(self, x, t):
        self.x = x
        self.t = t
        errors = (t - x) ** 2
        sums = np.sum(errors, axis=1)
        self.loss = np.mean(sums)
        return self.loss

    def backward(self, d_out=1):
        return -d_out * (self.t - self.x) * self.x / self.x.shape[0]
