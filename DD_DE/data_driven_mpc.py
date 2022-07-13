import numpy as np

from ortools.sat.python import cp_model

from DD_DE import helpers
from DD_DE import lti_system
from DD_DE import data_driven_predictor
from DD_DE import disturbance


class DataDrivenMPC:
    def __init__(self, input_sequence, state_sequence, prediction_horizon):
        self.dim_u = input_sequence.shape[0]
        self.dim_x = state_sequence.shape[0]

        # Todo: Make system flexibel

        # Todo: Load constraints from config.yaml
        # Hardcode constraints (for system with 2 inputs) for now
        G_u = np.diag(np.ones(2))
        G_u = np.concatenate([G_u, G_u])
        g_u = np.array([10, 10, -10, -10]).reshape(-1, 1)

        # Todo: Add constraints for states
        # Left out for now, because they have to be formulated in relation to u (extensive)

        # Matrices for input and state cost
        R = np.diag(np.ones(2))
        Q = np.diag(np.ones(2))

        self.h_matrix = helpers.create_hankel_matrix(
            input_sequence, state_sequence, prediction_horizon)
        self.h_matrix_inv = helpers.create_hankel_pseudo_inverse(
            self.h_matrix, self.dim_u, self.dim_x)

        u_sequence = np.array([1, 2, -1, -2, 3, 5])
        current_x = np.array([0, 1])
        self.predict_state_sequence(current_x, u_sequence)

    def get_new_u(self):
        # Todo: Später wirst du hier einen Zielstate als Input übergeben
        x = 1

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

        model = cp_model.CpModel()

        u_input = list()
        # u_squared = list()
        for i in range(0,6):
            u_input.append(model.NewIntVar(-5,5,"Entscheider "+str(i)))
            # u_squared.append(model.NewIntVar(-5,5,"Squared "+str(i)))

        # for i,u in u_input:
            # model.AddMultiplicationEquality(u_squared[i], [u, u])

        # Mach erstmal mit Einsernorm
        # self.get_state_sum()+
        model.Minimize(self.get_sum_states(u_input)+sum(u_input))

        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        
        print("Funktioniert bis hier")

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            print(f'Maximum of objective function: {solver.ObjectiveValue()}\n')
            # print(f'x = {solver.Value(x)}')
            # print(f'y = {solver.Value(y)}')
        else:
            print('No solution found.')


    def get_sum_states(self,u_sequence):
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

my_mpc = DataDrivenMPC(INPUT_SEQUENCE, state_sequence, 3)
print(my_mpc.h_matrix.shape)

my_mpc.get_new_u()


# Create LTI-System and read sequence


# my_first_mpc = DataDrivenMPC(input, state, 3)
