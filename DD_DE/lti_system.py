import numpy as np


class LTISystem:
    def __init__(self, x, A, B,get_dist):
        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float)
        self.x = np.array(x, dtype=float)
        self.get_dist = get_dist

        self.k=0

    def next_step(self, u):
        self.u = np.array(u)

        # Todo: When matrix multidimensional: * -> @
        self.x = self.A * self.x + self.B * self.u + self.get_dist()

        self.k += 1
