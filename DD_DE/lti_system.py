import numpy as np


class LTISystem:
    # Todo: Insert disturbance
    def __init__(self, x, A, B):
        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float)
        self.x = np.array(x, dtype=float)

    def next_step(self, u):
        self.u = np.array(u)
        self.x = self.A @ self.x + self.B @ self.u
