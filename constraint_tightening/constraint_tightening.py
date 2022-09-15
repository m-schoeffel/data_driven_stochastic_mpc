import numpy as np

from config import load_parameters


class ConstraintTightening:
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

        self.numbr_state_constr = self.G_u.shape[0]

    def tighten_constraints(self):
        x = 1

    def get_tightened_constraints(self):
        return self.G_v, self.g_v, self.G_z, self.g_z

    def tighten_constraints_on_interv(self, dist_interval):
        """Tighten constraints based on disturbance intervals"""

        # Todo: Create more flexible solution
        # Currently solution is hardcoded for systems with no joint constraints (Every state has it's own constraints)

        for i in range(0, self.numbr_state_constr):
            # Calculate dist interval for state, which corresponds to constraint
            # --> Only works, if there are no joint constraints
            interval = np.abs(self.G_x[i, :]) @ dist_interval
            if (np.sum(self.G_x) > 0):
                self.g_z[i] = self.g_x[i]+interval[0]
            else:
                self.g_z[i] = self.g_x[i]+interval[1]

        return self.G_v, self.g_v, self.G_z, self.g_z
