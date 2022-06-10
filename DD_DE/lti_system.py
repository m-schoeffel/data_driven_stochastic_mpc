import numpy as np


class LTISystem:
    def __init__(self, x, A, B, disturbance):
        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float)
        self.x = np.array(x, dtype=float)
        self.disturbance = disturbance

        print(f"self.A: \n{self.A}")
        print(f"self.B: \n{self.B}")
        print(f"self.x: \n{self.x}")

        self.k = 0

    def next_step(self, u, add_disturbance = True):
        self.u = np.array(u)

        # Todo: When matrix multidimensional: * -> @
        if add_disturbance:
            self.x = self.A @ self.x + self.B * self.u + np.array([0,self.disturbance.get_dist()]).reshape(-1,1)
        else:
            self.x = self.A @ self.x + self.B * self.u

        print(self.x)
        self.k += 1
        return self.x
