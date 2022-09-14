import numpy as np

from config import load_parameters

class ConstraintTighting:
    def __init__(self):
        constraints = load_parameters.load_constraints()
        self.G_u = np.array(constraints["G_u"])
        self.g_u = np.array(constraints["g_u"])
        self.G_x = np.array(constraints["G_x"])
        self.g_x = np.array(constraints["g_x"])

        # Initialize pseudo constraints
        self.G_v = self.G_u
        self.g_v = self.g_u
        self.G_z = self.G_x
        self.g_z = self.g_x

    def tighten_constraints(self):
        x = 1

    def get_tightened_constraints(self):
        return self.G_v,self.g_v,self.G_z,self.g_z