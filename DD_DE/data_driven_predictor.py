import numpy as np


class DDPredictor:
    def __init__(self, input_sequence, state_sequence):

        self.dim_u = input_sequence.shape[0]
        self.dim_x = state_sequence.shape[0]
        # self.h_matrix = np.array(h_matrix)
        # print(f"h_matrix (numpy) \n {self.h_matrix}")
        hankel_u_shape = (input_sequence.shape[0]*2, input_sequence.shape[1]-1)
        hankel_x_shape = (state_sequence.shape[0]*2, input_sequence.shape[1]-1)

        hankel_u = np.zeros(hankel_u_shape)
        hankel_x = np.zeros(hankel_x_shape)

        print(f"input_sequence:\n {input_sequence}")
        print(f"state_sequence:\n {state_sequence}")

        for i in range(0, input_sequence.shape[1]-1):
            hankel_u[:, i] = np.concatenate(
                (input_sequence[:, i], input_sequence[:, i+1]))
            hankel_x[:, i] = np.concatenate(
                (state_sequence[:, i], state_sequence[:, i+1]))

        # hankel_u = np.transpose(np.array(hankel_u))
        # hankel_x = np.transpose(np.array(hankel_x))

        print(f"hankel_u:\n{hankel_u}")
        print(f"hankel_x:\n{hankel_x}")

        self.h_matrix = np.concatenate((hankel_u, hankel_x))
        print(f"self.h_matrix:\n{self.h_matrix}")

        # Select relevant rows vor pseudo inverse of hankel matrix (used for prediction step)
        u_rows_idx = list(range(0, self.dim_u))
        x_rows_idx = list(range(self.dim_u*2, self.dim_u*2+self.dim_x))
        print(f"u_rows_idx: \n {u_rows_idx}")
        print(f"x_rows_idx: \n {x_rows_idx}")
        relevant_rows = u_rows_idx + x_rows_idx
        print(relevant_rows)

        h_input_state = self.h_matrix[relevant_rows,:]
        print(f"h_input_state (numpy) \n {h_input_state}")
        self.h_matrix_inv = np.linalg.pinv(h_input_state)

    # Todo: Currently for one dimensional state space, has to be changed for multidimensional state space

    def predict_state(self, current_x, u):
        current_x = np.array(current_x)
        u = np.array(u)

        goal_vector = np.vstack([u.reshape(-1,1), current_x.reshape(-1,1)])
        # print(goal_vector)
        alpha = self.h_matrix_inv @ goal_vector
        # Todo: Change name prediction (bad name)
        prediction = self.h_matrix @ alpha
        print(prediction)
        next_x = prediction[5]
        return next_x
