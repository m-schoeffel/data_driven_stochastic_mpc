import numpy as np


class DDPredictor:
    def __init__(self, h_matrix):
        self.h_matrix = np.array(h_matrix)
        self.h_matrix_inv = np.linalg.pinv(self.h_matrix)

    # Todo: Currently for one dimensional state space, has to be changed for multidimensional state space

    def predict_state(self,current_x,u):
        goal_vector = np.transpose(np.array([u,0,current_x,0]))
        print(f"goal vector: {goal_vector}")
        alpha = self.h_matrix_inv @ goal_vector
        print(f"alpha: {alpha}")
        # Todo: Change name prediction (bad name)
        prediction = self.h_matrix @ alpha
        print(f"Prediction: {prediction}")
        next_x = prediction[3]
        return next_x