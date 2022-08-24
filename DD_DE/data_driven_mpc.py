from re import U
import numpy as np
import time
import matplotlib.pyplot as plt

from scipy.optimize import minimize, LinearConstraint

from DD_DE import helpers
from DD_DE import lti_system
from DD_DE import data_driven_predictor
from DD_DE import disturbance


class DataDrivenMPC:
    def __init__(self, input_sequence, state_sequence):
        self.dim_u = input_sequence.shape[0]
        self.dim_x = state_sequence.shape[0]

        self.prediction_horizon = helpers.load_prediction_horizon()

        # Todo: Make system flexibel

        # Todo: Load constraints from config.yaml
        # Hardcode constraints (for system with 2 inputs) for now
        constraints = helpers.load_constraints()
        self.G_u = np.array(constraints["G_u"])
        self.g_u = np.array(constraints["g_u"])
        self.G_x = np.array(constraints["G_x"])
        self.g_x = np.array(constraints["g_x"])


        # Matrices for input and state cost
        cost_matrices = helpers.load_cost_matrices()
        self.R = np.array(cost_matrices["R"])
        self.Q = np.array(cost_matrices["Q"])

        self.h_matrix = helpers.create_hankel_matrix(
            input_sequence, state_sequence, self.prediction_horizon)
        self.h_matrix_inv = helpers.create_hankel_pseudo_inverse(
            self.h_matrix, self.dim_u, self.dim_x)

        u_sequence = np.array([1, 2, -1, -2, 3, 5])
        current_x = np.array([0, 1])
        # trajectory = self.predict_state_sequence(current_x, u_sequence)
        # cost = self.get_sequence_cost(u_sequence, current_x)
        # print(f"The sum of traj is {trajectory.transpose()@trajectory} and the cost is {cost}")

    def get_new_u(self, current_x):
        start_time = time.time()
        # Todo: Später wirst du hier einen Zielstate als Input übergeben

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
        # print(type(G_alpha),type(g_compl),sep="------------\n")

        # Make sure trajectory starts at current_x
        C_x_0 = np.zeros([len(current_x.reshape(-1, 1)), G_compl.shape[1]])
        for i in range(0, len(current_x.reshape(-1, 1))):
            C_x_0[i, self.dim_u*(self.prediction_horizon+1)+i] = 1
        C_x_0 = C_x_0 @ self.h_matrix

        print(current_x.reshape(-1, 1).shape)

        constr_x_0 = LinearConstraint(
            C_x_0, lb=current_x.reshape(-1,), ub=current_x.reshape(-1,))
        # constr_input_state,constr_x_0
        res = minimize(self.get_sequence_cost, np.zeros(
            self.h_matrix.shape[1]), args=(), constraints=[constr_input_state, constr_x_0])
        print(res)
        trajectory = (self.h_matrix @ res.x).reshape(-1, 1)
        print(trajectory)
        print(trajectory.shape)
        print(trajectory[-15])
        print("--- \"DataDrivenMPC.get_new_u\" took %s seconds ---" % (time.time() - start_time))

        # Plot for debugging
        fig, axs = plt.subplots(3, 2)

        x_coord = list(range(0,6))
        x_u = list(range(0,5))
        u_1 = [u for i,u in enumerate(trajectory[0:10]) if i%2==0]
        print(len(u_1))
        axs[0,0].plot(x_u,u_1,label="u_1")

        u_2 = [u for i,u in enumerate(trajectory[0:10]) if i%2==1]
        print(len(u_1))
        axs[0,1].plot(x_u,u_2,label="u_2")

        x_1 = [x for i,x in enumerate(trajectory[12:36]) if i%4==0]
        print(len(x_1))
        axs[1,0].plot(x_coord,x_1,label="x_1")

        x_2 = [x for i,x in enumerate(trajectory[12:36]) if i%4==1]
        print(len(x_2))
        axs[1,1].plot(x_coord,x_2,label="x_2")

        x_3 = [x for i,x in enumerate(trajectory[12:36]) if i%4==2]
        print(len(x_3))
        axs[2,0].plot(x_coord,x_3,label="x_3")

        x_4 = [x for i,x in enumerate(trajectory[12:36]) if i%4==3]
        print(len(x_4))
        axs[2,1].plot(x_coord,x_4,label="x_4")

        plt.show()
        print()


    def transform_state_constraints(self, G_x, g_x, current_x):
        """Transform state constraints to depend on u"""
        # This function is used to express the state constraints in dependance on u
        # So the control inputs can be the only decision variable and state constraints can still be used
        x = 1
        # G_x_u
        # g_x_u

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
            cost += trajectory[i:i+self.dim_x].transpose()@self.R@trajectory[i:i+self.dim_x]
        return cost

    def get_sum_states(self, u_sequence):
        print(f"\nget_sum_states:\n {u_sequence[0]}\n")
        return sum(u_sequence)

    def predict_state_sequence(self, current_x, u_sequence):
        current_x = np.array(current_x)
        u = np.array(u_sequence)

        goal_vector = np.vstack([u.reshape(-1, 1), current_x.reshape(-1, 1)])
        # print(goal_vector)
        alpha = self.h_matrix_inv @ goal_vector
        trajectory = self.h_matrix @ alpha
        # # print(prediction)
        # indices_of_prediction = list(
        #     range(self.dim_u*2+self.dim_x, self.dim_u*2+self.dim_x*2))
        # # print(f"indices_of_prediction: {indices_of_prediction}")
        # next_x = trajectory[indices_of_prediction]
        # return next_x
        # print(f"\n\ntrajectory: {np.round(trajectory).astype(np.int)}\n\n")
        # print(f"\n\ntrajectory: {trajectory}\n\n")
        return trajectory


# Testbench:
[main_param, lti_system_param] = helpers.load_parameters()

NUMBER_OF_MEASUREMENTS = main_param["number_of_measurements"]

# gaussian_process/traditional_kde/discounted_kde
DISTURBANCE_ESTIMATION = main_param["dist_est"]

# Specify the type of disturbance for each state
# gaussian/uniform/triangular/lognormal
TYPES_OF_DISTURBANCES = lti_system_param["dist"]


A_SYSTEM_MATRIX = lti_system_param["A"]
B_INPUT_MATRIX = lti_system_param["B"]

X_INITIAL_STATE = lti_system_param["x_0"]

INPUT_SEQUENCE = main_param["input_seq"]

my_disturbance = disturbance.Disturbance(TYPES_OF_DISTURBANCES)

my_system = lti_system.LTISystem(
    x=X_INITIAL_STATE, A=A_SYSTEM_MATRIX, B=B_INPUT_MATRIX, disturbances=my_disturbance)

state_sequence = np.zeros(
    (X_INITIAL_STATE.shape[0], INPUT_SEQUENCE.shape[1]+1))
state_sequence[:, 0] = X_INITIAL_STATE[:, 0]
# Record input-state sequence
for i in range(INPUT_SEQUENCE.shape[1]):
    state_sequence[:, i+1] = my_system.next_step(
        INPUT_SEQUENCE[:, i], add_disturbance=False)[:, 0]

my_mpc = DataDrivenMPC(INPUT_SEQUENCE, state_sequence)
print(my_mpc.h_matrix.shape)

my_mpc.get_new_u(np.array([2, 2,-2,-2]))


# Create LTI-System and read sequence


# my_first_mpc = DataDrivenMPC(input, state, 3)
