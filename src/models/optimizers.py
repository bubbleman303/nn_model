from typing import List
import numpy as np

from src.models import layers


class SGD:
    def __init__(self, affine_list: List[layers.AffineLayer], lr=0.3):
        self.affine_list = affine_list
        self.lr = lr

    def update(self):
        for affine in self.affine_list:
            affine.w -= self.lr * affine.dw
            affine.b -= self.lr * affine.db


class AdaGrad:
    def __init__(self, affine_list: List[layers.AffineLayer], lr=0.3):
        self.affine_list = affine_list
        self.lr = lr
        self.h = None

    def update(self):
        if self.h is None:
            self.h = []
            for affine in self.affine_list:
                self.h.append([np.zeros_like(affine.w), np.zeros_like(affine.b)])

        for index, affine in enumerate(self.affine_list):
            self.h[index][0] += affine.dw ** 2
            affine.w -= self.lr * affine.dw / (np.sqrt(self.h[index][0]) + 1e-7)
            self.h[index][1] += affine.db ** 2
            affine.b -= self.lr * affine.db / (np.sqrt(self.h[index][1]) + 1e-7)


class Momentum:
    def __init__(self, affine_list: List[layers.AffineLayer], lr=0.3):
        self.affine_list = affine_list
        self.momentum = 0.9
        self.lr = lr
        self.v = None

    def update(self):
        if self.v is None:
            self.v = []
            for affine in self.affine_list:
                self.v.append([np.zeros_like(affine.w), np.zeros_like(affine.b)])

        for index, affine in enumerate(self.affine_list):
            self.v[index][0] = self.momentum * self.v[index][0] - self.lr * affine.dw
            affine.w += self.v[index][0]
            self.v[index][1] = self.momentum * self.v[index][1] - self.lr * affine.db
            affine.b += self.v[index][1]
