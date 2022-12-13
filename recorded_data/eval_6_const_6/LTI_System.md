The LTI-System during recording this dataset:

import numpy as np


class LTISystem:
    def __init__(self, x, A, B, dist_seq):
        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float)
        self.x = np.array(x, dtype=float)
        self.dist_seq = dist_seq

        self.k = 0

    def next_step(self, u, add_disturbance=True, k=None):
        self.u = np.array(u).reshape(-1, 1)

        if add_disturbance and k != None:
            self.x = self.A @ self.x + self.B @ self.u + \
                self.dist_seq[:, k].reshape(-1, 1) + np.array([[-0.1,0,0,0],[0,-0.1,0,0],[0,0,0,0],[0,0,0,0]])@self.x
        else:
            self.x = self.A @ self.x + self.B @ self.u

        # print(self.x)
        self.k += 1
        return self.x
