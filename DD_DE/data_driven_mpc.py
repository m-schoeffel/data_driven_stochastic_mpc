import numpy as np


class DataDrivenMPC:
    def __init__(self, input_sequence, state_sequence, prediction_horizon):
        n = prediction_horizon

        self.dim_u = input_sequence.shape[0]
        self.dim_x = state_sequence.shape[0]

        hankel_u_shape = (
            input_sequence.shape[0]*(n+1), input_sequence.shape[1]-n)
        hankel_x_shape = (
            state_sequence.shape[0]*(n+1), input_sequence.shape[1]-n)

        hankel_u = np.zeros(hankel_u_shape)
        hankel_x = np.zeros(hankel_x_shape)

        # print(f"input_sequence:\n {input_sequence}")
        # print(f"state_sequence:\n {state_sequence}")

        # Split input and state sequences horizontally to efficiently concatenate them for varying prediction horizons
        input_sequence_splitted = np.hsplit(
            input_sequence, input_sequence.shape[1])
        state_sequence_splitted = np.hsplit(
            state_sequence, state_sequence.shape[1])

        for i in range(0, input_sequence.shape[1]-n):
            hankel_u[:, i] = np.concatenate(
                tuple(input_sequence_splitted[idx] for idx in range(i, i+n+1)))[:,0]
            hankel_x[:, i] = np.concatenate(
                tuple(state_sequence_splitted[idx] for idx in range(i, i+n+1)))[:,0]
        
        # print(f"\nhankel_u: \n{hankel_u}")
        # print(f"\nhankel_x: \n{hankel_x}")

        self.h_matrix = np.concatenate((hankel_u, hankel_x))

        print(self.h_matrix.shape)


input = np.array([[1, -1, 0, 2, 3, -4, 0, -6, 2],
                 [-1, 0, 2, 3, -4, 0, -6, 2, 4]])
state = np.array([[1, -1, 0, 2, 3, -4, 0, -6, 2],
                 [-1, 0, 2, 3, -4, 0, -6, 2, 4]])

my_first_mpc = DataDrivenMPC(input, state, 3)
