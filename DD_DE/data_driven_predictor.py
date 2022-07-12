import numpy as np

from DD_DE import helpers


class DDPredictor:
    def __init__(self, input_sequence, state_sequence):

        self.dim_u = input_sequence.shape[0]
        self.dim_x = state_sequence.shape[0]

        self.h_matrix = helpers.create_hankel_matrix(
            input_sequence, state_sequence, prediction_horizon=1)

        print(f"self.h_matrix:\n{self.h_matrix}")

        # Select relevant rows vor pseudo inverse of hankel matrix (used for prediction step)
        u_rows_idx = list(range(0, self.dim_u))
        x_rows_idx = list(range(self.dim_u*2, self.dim_u*2+self.dim_x))
        print(f"u_rows_idx: \n {u_rows_idx}")
        print(f"x_rows_idx: \n {x_rows_idx}")
        relevant_rows = u_rows_idx + x_rows_idx
        print(relevant_rows)

        h_input_state = self.h_matrix[relevant_rows, :]
        print(f"h_input_state (numpy) \n {h_input_state}")
        self.h_matrix_inv = np.linalg.pinv(h_input_state)

    # Todo: Currently for one dimensional state space, has to be changed for multidimensional state space

    def predict_state(self, current_x, u):
        current_x = np.array(current_x)
        u = np.array(u)

        goal_vector = np.vstack([u.reshape(-1, 1), current_x.reshape(-1, 1)])
        # print(goal_vector)
        alpha = self.h_matrix_inv @ goal_vector
        trajectory = self.h_matrix @ alpha
        # print(prediction)
        indices_of_prediction = list(
            range(self.dim_u*2+self.dim_x, self.dim_u*2+self.dim_x*2))
        # print(f"indices_of_prediction: {indices_of_prediction}")
        next_x = trajectory[indices_of_prediction]
        return next_x
