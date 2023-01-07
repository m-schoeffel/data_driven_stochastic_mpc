import os
import numpy as np
import time

from scipy.optimize import minimize, LinearConstraint
from sklearn.metrics import recall_score

from . import _hankel_helpers


class DataDrivenMPC:
    def __init__(self, input_sequence, state_sequence, predic_hori_size, state_cost, input_cost, record_data=False, folder_name=""):
        self.dim_u = input_sequence.shape[0]
        self.dim_x = state_sequence.shape[0]

        # Set size of the prediction horizon
        self.predic_hori_size = predic_hori_size

        # Matrices for state and input cost
        self.R = np.array(state_cost)
        self.Q = np.array(input_cost)

        self.h_matrix = _hankel_helpers.create_hankel_matrix(
            input_sequence, state_sequence, self.predic_hori_size)
        self.h_matrix_inv = _hankel_helpers.create_hankel_pseudo_inverse(
            self.h_matrix, self.dim_u, self.dim_x)

        # Create folders to store data
        self.record_data = record_data
        if self.record_data:
            path_root_folder = os.getcwd()
            record_folder_path = os.path.join(path_root_folder, "recorded_data", folder_name, "mpc")
            os.mkdir(record_folder_path)
            
            self.trajectory_path = os.path.join(record_folder_path,"optimal_trajectories")
            os.mkdir(self.trajectory_path)

            self.states_path = os.path.join(record_folder_path,"measured_states")
            os.mkdir(self.states_path)

    def get_new_u(self, current_x, G_v, g_v, G_z, g_z, k, ref_pred_hor=0,exp_x=None):

        # Specify ref_pred_hor
        if type(ref_pred_hor) == int and ref_pred_hor == 0:
            self.ref_pred_hor = np.zeros([self.dim_x, self.predic_hori_size])
        else:
            self.ref_pred_hor = np.array(ref_pred_hor)

        # Adjust reference trajectory horizon to account for expected value of disturbance
        # Improves performance
        for i in range(0,self.predic_hori_size):
            self.ref_pred_hor[:,i] = self.ref_pred_hor[:,i].reshape(-1)-exp_x.reshape(-1)*(i+1)

        self.len_g_z = g_z.shape[0]

        # Get constraint matrices to cover full sequence (fs) of input and state
        G_v_fs = self.determine_full_seq_constr_matrix(G_v)
        g_v_fs = self.determine_full_seq_constr_ub(g_v)
        G_z_fs = self.determine_full_seq_constr_matrix(G_z)
        g_z_fs = self.determine_full_seq_constr_ub(g_z)
        G_compl = self.determine_complete_constraint_matrix(G_v_fs, G_z_fs)

        # Adjust upperbound of state constraint to account for expected value of disturbance
        g_z_fs = self.adjust_state_constraint_upper_bound(G_z,g_z_fs,exp_x)

        # Create Constraintmatrix, which is depending on alpha (decision variable in solver)
        G_alpha = G_compl @ self.h_matrix

        # Make sure trajectory starts at current_x
        C_x_0 = np.zeros([len(current_x.reshape(-1, 1)), G_compl.shape[1]])
        for i in range(0, len(current_x.reshape(-1, 1))):
            C_x_0[i, self.dim_u*(self.predic_hori_size+1)+i] = 1
        C_x_0 = C_x_0 @ self.h_matrix

        constr_x_0 = LinearConstraint(
            C_x_0, lb=current_x.reshape(-1,), ub=current_x.reshape(-1,))

        # Use feasible starting point for optimization (In this case u=0 for inputs and x_0 for initial state)
        # Needed, because scipy.minimize() produces faulty result otherwise
        starting_point_opt = np.vstack(
            [np.ones([self.dim_u*(self.predic_hori_size), 1]), current_x])
        alpha_0 = self.h_matrix_inv@starting_point_opt

        # Try to find optimal feasible trajectory
        # If no feasible trajectory is found, lift the state constraints for #nmbr_lift_constraints states, beginning with the initial state
        for nmbr_lift_constraints in range(0, self.predic_hori_size+2):
            res = self.optimize_cost_over_constraints(
                alpha_0, G_alpha, g_v_fs, g_z_fs, constr_x_0, nmbr_lift_constraints)
            if res.success:
                break

        trajectory = (self.h_matrix @ res.x).reshape(-1, 1)

        if self.record_data:
            filename_traj = os.path.join(self.trajectory_path,"trajectory_k_"+str(k))
            np.save(filename_traj,trajectory)

            filename_state = os.path.join(self.states_path,"real_state_k_"+str(k))
            np.save(filename_state,current_x)

        # return next input (MPC Ouput) and predicted next state
        next_u = trajectory[0:self.dim_u]
        x_pred = trajectory[self.dim_u*(self.predic_hori_size+1) +
                            self.dim_x:self.dim_u*(self.predic_hori_size+1)+self.dim_x*2]

        # Return prediction horizon for later visualization
        prediction_horizon = trajectory[self.dim_u*(self.predic_hori_size+1)+self.dim_x:self.dim_u*(
            self.predic_hori_size+1)+self.dim_x*(1+self.predic_hori_size)]
        return next_u, x_pred, prediction_horizon

    def optimize_cost_over_constraints(self, alpha_0, G_alpha, g_v_fs, g_z_fs, constr_x_0, nmbr_lift_constraints=0):
        """This function implements the solver of the mpc module
            If no feasible solution can be found given the system dynamics and state constraints, the state constraints on the prediction horizon are lifted one by one
            The parameter nmbr_lift_constraints specifies the number of states in the prediction horizon which do not have to satisfy the constraints"""

        # Lift upper bound for first #nmbr_lift_constraints states to get feasible trajectory
        g_z_fs[0:nmbr_lift_constraints*self.len_g_z] = np.inf

        g_compl = np.vstack([g_v_fs, g_z_fs])
        g_compl = g_compl.reshape(-1,)

        # Create Alpha Constraints

        constr_input_state = LinearConstraint(
            G_alpha, lb=-np.ones(g_compl.shape)*np.inf, ub=g_compl)

        # optimization, if trajectory is noise free
        res = minimize(self.get_sequence_cost, alpha_0, args=(
        ), method='SLSQP',constraints=[constr_input_state, constr_x_0])

        # # optimization, if trajectory is corrupted by measurement noise
        # res = minimize(self.get_sequence_cost, alpha_0, args=(
        # ), method='SLSQP',tol=0.01, constraints=[constr_input_state, constr_x_0])

        return res

    def determine_complete_constraint_matrix(self, G_v_fs, G_z_fs):
        u_block = np.hstack(
            [G_v_fs, np.zeros([G_v_fs.shape[0], G_z_fs.shape[1]])])
        x_block = np.hstack(
            [np.zeros([G_z_fs.shape[0], G_v_fs.shape[1]]), G_z_fs])
        G_compl = np.vstack([u_block, x_block])
        return G_compl

    def determine_full_seq_constr_matrix(self, matrix):
        """Scale up constraint matrices to cover all states and inputs in prediction horizon"""

        steps = self.predic_hori_size

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
        """Scale up constraint upper bound vectors to cover all states and inputs in prediction horizon"""

        steps = self.predic_hori_size

        ub_fs = np.tile(upper_bound, [steps+1, 1])

        return ub_fs

    def get_sequence_cost(self, alpha):

        trajectory = self.h_matrix @ alpha
        cost = 0

        # Input cost is the the sum of inputs squared and weighted by the cost matrix Q
        for i in range(0, self.dim_u*self.predic_hori_size, self.dim_u):
            cost += trajectory[i:i +
                               self.dim_u].transpose()@self.Q@trajectory[i:i+self.dim_u]

        # State cost is the quadratic difference between the reference trajectory for the prediction horizon and the actual prediction, weighted with the cost matrix R
        for i in range(0, self.predic_hori_size):
            idx_state = self.dim_u * \
                (self.predic_hori_size+1)+self.dim_x + i*self.dim_x
            state_pred = trajectory[idx_state:idx_state+self.dim_x]
            state_ref = self.ref_pred_hor[:, i]
            state_diff = state_pred - state_ref
            cost += state_diff.transpose()@self.R@state_diff

        # Add regularization of alpha in case of measurement noise
        # cost += np.dot(alpha,alpha)*10000

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

    def adjust_state_constraint_upper_bound(self,G_z,g_z_fs,exp_x):
        """This function adjusts the upper bound of the state constraints to account for the expected value of the disturbance
            This step improves the tracking performance but can potentially lead to more frequent constraint violations while the underlying distribution of the disturbance changes"""
        
        num_constr = G_z.shape[0]
        lin_com_dist = G_z @ exp_x
        for i in range(1,self.predic_hori_size+1):
            g_z_fs[i*num_constr:(i+1)*num_constr] -= lin_com_dist*1*i

        return g_z_fs
