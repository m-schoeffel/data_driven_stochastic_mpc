import numpy as np

from DD_DE import helpers


class DataDrivenMPC:
    def __init__(self, input_sequence, state_sequence, prediction_horizon):
        self.dim_u = input_sequence.shape[0]
        self.dim_x = state_sequence.shape[0]



        # Todo: Make system flexibel

        # Todo: Load constraints from config.yaml
        # Hardcode constraints (for system with 2 inputs) for now
        G_u = np.diag(np.ones(2))
        G_u = np.concatenate([G_u,G_u])
        g_u = np.array([10,10,-10,-10]).reshape(-1,1)

        # Todo: Add constraints for states
        # Left out for now, because they have to be formulated in relation to u (extensive)

        # Matrices for input and state cost
        R = np.diag(np.ones(2))
        Q = np.diag(np.ones(2))


        self.h_matrix = helpers.create_hankel_matrix(input_sequence,state_sequence,prediction_horizon)



    def get_new_u():
        # Todo: Später wirst du hier einen Zielstate als Input übergeben
        x=1

        # Hier muss das OR Modell geladen werden
        
        # Dazu gehören...
        # 1. Deklarierung der Entscheidungsvariablen (In deinem Fall die Us)
        # 2. Festlegung der Constraints (Anhand der derzeitigen Constraintmatrizen aka G_u und G_x)
        # 3. Festlegung der Optimierungsfunktion

        # Nötige Hilfsfunktionen:
        # 1. Funktion, die Us annimmt, und x_1, x_2, usw. zurückgibt
        # 2. Funktion, die quadrierte Variablen aufspannt für Optimierungsfunktion
        # 3. Funktion, die Optimierungsfunktion anhand der Matrizen als Einzeiler zurückgibt



# Just for debugging purposes:
input = np.array([[1, -1, 0, 2, 3, -4, 0, -6, 2],
                 [-1, 0, 2, 3, -4, 0, -6, 2, 4]])


# Create LTI-System and read sequence

# my_first_mpc = DataDrivenMPC(input, state, 3)
