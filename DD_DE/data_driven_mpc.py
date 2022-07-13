import numpy as np

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

        # Todo: Add constraints for states
        # Left out for now, because they have to be formulated in relation to u (extensive)

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
        trajectory=self.predict_state_sequence(current_x, u_sequence)
        cost = self.get_sequence_cost(u_sequence,current_x)
        print(f"The sum of traj is {trajectory.transpose()@trajectory} and the cost is {cost}")

    def get_new_u(self):
        # Todo: Später wirst du hier einen Zielstate als Input übergeben
        
        # Scale constraints to cover full input and state sequence
        # Todo: Ich hardcode jetzt das System mit festen Dimensionen, muss später variabel gemacht werden

        # Hier muss das OR Modell geladen werden

        # Dazu gehören...
        # 1. Deklarierung der Entscheidungsvariablen (In deinem Fall die Us)
        # 2. Festlegung der Constraints (Anhand der derzeitigen Constraintmatrizen aka G_u und G_x)
        # 3. Festlegung der Optimierungsfunktion

        # Nötige Hilfsfunktionen:
        # 1. Funktion, die Us annimmt, und x_1, x_2, usw. zurückgibt
        # 2. Funktion, die quadrierte Variablen aufspannt für Optimierungsfunktion
        # 3. Funktion, die die Optimierungsfunktion anhand der Matrizen als Einzeiler zurückgibt

        # Create U Constraints
        constraint = LinearConstraint(
            self.G_u, lb=-self.g_u, ub=self.g_u)
        
        res = minimize(self.get_sequence_cost,[0,0,0,0,0,0],args=(np.array([10,13])),constraints=constraint)
        print(res)
        print(self.predict_state_sequence(np.array([10,13]),res.x))

    def get_sequence_cost(self, u_seq, current_x):
        trajectory = self.predict_state_sequence(current_x, u_seq)
        cost = 0
        for i in [0, 2, 4]:
            cost += trajectory[i:i+2].transpose()@self.Q@trajectory[i:i+2]

        for i in [8, 10, 12, 14]:
            cost += trajectory[i:i+2].transpose()@self.R@trajectory[i:i+2]
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
        print(f"\n\ntrajectory: {np.round(trajectory).astype(np.int)}\n\n")
        # print(f"\n\ntrajectory: {trajectory}\n\n")
        return trajectory


# Testbench:
[main_param, lti_system_param, disc_kde_param] = helpers.load_parameters()

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

my_mpc.get_new_u()


# Create LTI-System and read sequence


# my_first_mpc = DataDrivenMPC(input, state, 3)
