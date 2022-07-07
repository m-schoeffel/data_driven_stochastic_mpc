import numpy as np

from DD_DE import helpers


class DataDrivenMPC:
    def __init__(self, input_sequence, state_sequence, prediction_horizon):

        self.h_matrix = helpers.create_hankel_matrix(input_sequence,state_sequence,prediction_horizon)

        print(self.h_matrix)


input = np.array([[1, -1, 0, 2, 3, -4, 0, -6, 2],
                 [-1, 0, 2, 3, -4, 0, -6, 2, 4]])
state = np.array([[1, -1, 0, 2, 3, -4, 0, -6, 2],
                 [-1, 0, 2, 3, -4, 0, -6, 2, 4]])

my_first_mpc = DataDrivenMPC(input, state, 3)
