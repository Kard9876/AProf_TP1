import numpy as np

from Code.DNN.optimizations.regulator import Regulator


class L2Reg(Regulator):
    def __init__(self, l2_val):
        super().__init__()
        self._val = l2_val

    def update(self, n, w):
        return (self._val/(2*n)) * np.sum(w ** 2)
