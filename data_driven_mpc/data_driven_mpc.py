import numpy as np
import time

from scipy.optimize import minimize, LinearConstraint

from . import hankel_helpers
from config import load_parameters

class DataDrivenMPC:
    def __init__(self, input_sequence, state_sequence):
        self.dim_u = input_sequence.shape[0]
        self.dim_x = state_sequence.shape[0]

        self.prediction_horizon = load_parameters.load_prediction_horizon()

        # Todo: Make system flexibel

        # Hardcode constraints (for system with 2 inputs) for now
        constraints = load_parameters.load_constraints()
        self.G_u = np.array(constraints["G_u"])
        self.g_u = np.array(constraints["g_u"])
        self.G_x = np.array(constraints["G_x"])
        self.g_x = np.array(constraints["g_x"])


        # Matrices for input and state cost
        cost_matrices = load_parameters.load_cost_matrices()
        self.R = np.array(cost_matrices["R"])
        self.Q = np.array(cost_matrices["Q"])

        self.h_matrix = hankel_helpers.create_hankel_matrix(
            input_sequence, state_sequence, self.prediction_horizon)
        self.h_matrix_inv = hankel_helpers.create_hankel_pseudo_inverse(
            self.h_matrix, self.dim_u, self.dim_x)

    def get_new_u(self, current_x,goal_state=0):
        start_time = time.time()

        # Specify goal_state
        if goal_state == 0:
            self.goal_state = np.zeros([self.dim_x])
        else:
            self.goal_state = np.array(goal_state)

        # Get constraint matrices to cover full sequence (fs) of input and state
        G_u_fs = self.determine_full_seq_constr_matrix(self.G_u)
        g_u_fs = self.determine_full_seq_constr_ub(self.g_u)
        G_x_fs = self.determine_full_seq_constr_matrix(self.G_x)
        g_x_fs = self.determine_full_seq_constr_ub(self.g_x)
        G_compl = self.determine_complete_constraint_matrix(G_u_fs, G_x_fs)
        g_compl = np.vstack([g_u_fs, g_x_fs])
        g_compl = g_compl.reshape(-1,)

        # Create Constraintmatrix, which is depending on alpha (decision variable)
        G_alpha = G_compl @ self.h_matrix

        # Create Alpha Constraints
        constr_input_state = LinearConstraint(
            G_alpha, lb=-g_compl*np.inf, ub=g_compl)

        # Make sure trajectory starts at current_x
        C_x_0 = np.zeros([len(current_x.reshape(-1, 1)), G_compl.shape[1]])
        for i in range(0, len(current_x.reshape(-1, 1))):
            C_x_0[i, self.dim_u*(self.prediction_horizon+1)+i] = 1
        C_x_0 = C_x_0 @ self.h_matrix

        constr_x_0 = LinearConstraint(
            C_x_0, lb=current_x.reshape(-1,), ub=current_x.reshape(-1,))
        
        # Calculate feasible starting point for optimization
        # Needed, because scipy.minimize() produces faulty result otherwise
        alpha_0=self.h_matrix_inv@np.array([0,0,0,0,0,0,0,0,0,0,current_x[0],current_x[1],current_x[2],current_x[3]]).transpose()
        
        # constr_input_state,constr_x_0
        res = minimize(self.get_sequence_cost, alpha_0, args=(), method='SLSQP',constraints=[constr_input_state,constr_x_0])
        # print(res)
        trajectory = (self.h_matrix @ res.x).reshape(-1, 1)

        # return next input (MPC Ouput)
        # Todo: replace hardcoded indices by flexible selection
        # print(trajectory[0:2])
        next_u = trajectory[0:2]
        return next_u

    def determine_complete_constraint_matrix(self, G_u_fs, G_x_fs):
        u_block = np.hstack(
            [G_u_fs, np.zeros([G_u_fs.shape[0], G_x_fs.shape[1]])])
        x_block = np.hstack(
            [np.zeros([G_x_fs.shape[0], G_u_fs.shape[1]]), G_x_fs])
        G_compl = np.vstack([u_block, x_block])
        return G_compl

    def determine_full_seq_constr_matrix(self, matrix):
        """Scale up constraint matrices to cover full sequence"""

        steps = self.prediction_horizon

        full_constrainted_matrix = np.array([0, 0])

        full_constrainted_matrix = np.hstack(
            [matrix, np.tile(np.zeros(matrix.shape), steps)])

        for i in range(1, steps):
            left_side_row = np.tile(np.zeros(matrix.shape), i)
            right_side_row = np.tile(np.zeros(matrix.shape), steps-i)
            row = np.hstack([left_side_row, matrix, right_side_row])
            full_constrainted_matrix = np.vstack(
                [full_constrainted_matrix, row])

        row = np.hstack([np.tile(np.zeros(matrix.shape), steps), matrix])
        full_constrainted_matrix = np.vstack([full_constrainted_matrix, row])

        return full_constrainted_matrix

    def determine_full_seq_constr_ub(self, upper_bound):
        """Scale up constraint upper bound vectors to cover full sequence"""

        steps = self.prediction_horizon

        ub_fs = np.tile(upper_bound, [steps+1, 1])

        return ub_fs

    def get_sequence_cost(self, alpha):

        trajectory = self.h_matrix @ alpha
        cost = 0
        for i in range(0,self.dim_u*self.prediction_horizon,self.dim_u):
            cost += trajectory[i:i+self.dim_u].transpose()@self.Q@trajectory[i:i+self.dim_u]

        for i in range(self.dim_u*(self.prediction_horizon+1),self.dim_u*(self.prediction_horizon+1)+self.dim_x*(self.prediction_horizon+1),self.dim_x):
            state_diff = trajectory[i:i+self.dim_x] - self.goal_state
            cost += state_diff.transpose()@self.R@state_diff
        return cost

    def get_sum_states(self, u_sequence):
        print(f"\nget_sum_states:\n {u_sequence[0]}\n")
        return sum(u_sequence)

    def predict_state_sequence(self, current_x, u_sequence):
        current_x = np.array(current_x)
        u = np.array(u_sequence)

        goal_vector = np.vstack([u.reshape(-1, 1), current_x.reshape(-1, 1)])
        alpha = self.h_matrix_inv @ goal_vector
        trajectory = self.h_matrix @ alpha
        return trajectory