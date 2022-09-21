import numpy as np


class ConstraintTightening:
    def __init__(self,G_u,g_u,G_x,g_x):
        self.G_u = np.array(G_u)
        self.g_u = np.array(g_u)
        self.G_x = np.array(G_x)
        self.g_x = np.array(g_x)

        # Initialize pseudo constraints
        self.G_v = self.G_u.copy()
        self.g_v = self.g_u.copy()
        self.G_z = self.G_x.copy()
        self.g_z = self.g_x.copy()

        self.numbr_state_constr = self.G_x.shape[0]

    def tighten_constraints(self):
        x = 1

    def get_tightened_constraints(self):
        return self.G_v, self.g_v, self.G_z, self.g_z

    def tighten_constraints_on_interv(self, dist_interval):
        """Tighten constraints based on disturbance intervals"""

        # Todo: Create more flexible solution
        # Currently solution is hardcoded for systems with no joint constraints (Every state has it's own constraints)
        # No joint constraints means, that G_x is of the form G_x = [[1,0,0],[0,1,0,[1,0,0],[0,0,1]]]

        for i in range(0, self.numbr_state_constr):
            # Calculate dist interval for state, which corresponds to constraint
            # --> Only works, if there are no joint constraints
            interval = np.abs(self.G_x[i, :]) @ dist_interval
            if (np.sum(self.G_x[i]) > 0):
                self.g_z[i] = self.g_x[i]+interval[0]
            else:
                self.g_z[i] = self.g_x[i]+interval[1]

        return self.G_v, self.g_v, self.G_z, self.g_z
