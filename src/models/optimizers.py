from typing import List

from src.models import layers


class SGD:
    def __init__(self, affine_list: List[layers.AffineLayer], lr=0.3):
        self.affine_list = affine_list
        self.lr = lr

    def update(self):
        for affine in self.affine_list:
            affine.w -= self.lr * affine.dw
            affine.b -= self.lr * affine.db
