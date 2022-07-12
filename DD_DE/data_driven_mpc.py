import numpy as np

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

    def get_new_u():
        # Todo: Später wirst du hier einen Zielstate als Input übergeben
        x = 1

        # Hier muss das OR Modell geladen werden

        # Dazu gehören...
        # 1. Deklarierung der Entscheidungsvariablen (In deinem Fall die Us)
        # 2. Festlegung der Constraints (Anhand der derzeitigen Constraintmatrizen aka G_u und G_x)
        # 3. Festlegung der Optimierungsfunktion

        # Nötige Hilfsfunktionen:
        # 1. Funktion, die Us annimmt, und x_1, x_2, usw. zurückgibt
        # 2. Funktion, die quadrierte Variablen aufspannt für Optimierungsfunktion
        # 3. Funktion, die Optimierungsfunktion anhand der Matrizen als Einzeiler zurückgibt


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

my_mpc = DataDrivenMPC(INPUT_SEQUENCE,state_sequence,3)
print(my_mpc.h_matrix)


# Create LTI-System and read sequence


# my_first_mpc = DataDrivenMPC(input, state, 3)
