import numpy as np


class LTISystem:
    def __init__(self, x, A, B, disturbances):
        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float)
        self.x = np.array(x, dtype=float)
        self.disturbances = disturbances

        print(f"self.A: \n{self.A}")
        print(f"self.B: \n{self.B}")
        print(f"self.x: \n{self.x}")

        self.k = 0

    def next_step(self, u, add_disturbance = True):
        self.u = np.array(u)

        # Todo: When matrix multidimensional: * -> @
        if add_disturbance:
            self.x = self.A @ self.x + self.B * self.u + self.disturbances.get_dist_vector()
        else:
            self.x = self.A @ self.x + self.B * self.u

        # print(self.x)
        self.k += 1
        return self.x
