import numpy as np

from . import hankel_helpers


class DDPredictor:
    def __init__(self, input_sequence, state_sequence):

        self.dim_u = input_sequence.shape[0]
        self.dim_x = state_sequence.shape[0]

        self.h_matrix = hankel_helpers.create_hankel_matrix(
            input_sequence, state_sequence, prediction_horizon=1)

        # Select relevant rows vor pseudo inverse of hankel matrix (used for prediction step)
        u_rows_idx = list(range(0, self.dim_u))
        x_rows_idx = list(range(self.dim_u*2, self.dim_u*2+self.dim_x))
        relevant_rows = u_rows_idx + x_rows_idx

        h_input_state = self.h_matrix[relevant_rows, :]
        self.h_matrix_inv = np.linalg.pinv(h_input_state)

    def predict_state(self, current_x, u):
        current_x = np.array(current_x)
        u = np.array(u)

        goal_vector = np.vstack([u.reshape(-1, 1), current_x.reshape(-1, 1)])
        alpha = self.h_matrix_inv @ goal_vector
        trajectory = self.h_matrix @ alpha
        indices_of_prediction = list(
            range(self.dim_u*2+self.dim_x, self.dim_u*2+self.dim_x*2))
        next_x = trajectory[indices_of_prediction]
        return next_x
