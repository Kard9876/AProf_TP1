import numpy as np

from Code.DNN.optimizations.regulator import Regulator


class L1Reg(Regulator):
    def __init__(self, l1_val):
        super().__init__()
        self._val = l1_val

    def update(self, n, w):
        return (self._val/(2*n)) * np.sum(np.abs(w))
