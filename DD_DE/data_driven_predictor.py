import numpy as np


class DDPredictor:
    def __init__(self, h_matrix):
        self.h_matrix = np.array(h_matrix)
        print(f"h_matrix (numpy) \n {self.h_matrix}")

        h_input_state = self.h_matrix[[0,2],:]
        print(f"h_input_state (numpy) \n {h_input_state}")
        self.h_matrix_inv = np.linalg.pinv(h_input_state)

    # Todo: Currently for one dimensional state space, has to be changed for multidimensional state space

    def predict_state(self,current_x,u):
        goal_vector = np.transpose(np.array([u,current_x]))
        alpha = self.h_matrix_inv @ goal_vector
        # Todo: Change name prediction (bad name)
        prediction = self.h_matrix @ alpha
        next_x = prediction[3]
        return next_x