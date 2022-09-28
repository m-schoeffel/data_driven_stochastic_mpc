import numpy as np


def create_hankel_matrix(input_sequence, state_sequence, predic_hori_size):
    n = predic_hori_size

    dim_u = input_sequence.shape[0]
    dim_x = state_sequence.shape[0]

    length_hankel_matrix = input_sequence.shape[1]-n

    hankel_u_shape = (
        dim_u*(n+1), length_hankel_matrix)
    hankel_x_shape = (
        dim_x*(n+1), length_hankel_matrix)

    hankel_u = np.zeros(hankel_u_shape)
    hankel_x = np.zeros(hankel_x_shape)

    # Split input and state sequences horizontally to efficiently concatenate them for varying prediction horizons
    input_sequence_splitted = np.hsplit(
        input_sequence, input_sequence.shape[1])
    state_sequence_splitted = np.hsplit(
        state_sequence, state_sequence.shape[1])

    for i in range(0, length_hankel_matrix):
        hankel_u[:, i] = np.concatenate(
            tuple(input_sequence_splitted[idx] for idx in range(i, i+n+1)))[:, 0]
        hankel_x[:, i] = np.concatenate(
            tuple(state_sequence_splitted[idx] for idx in range(i, i+n+1)))[:, 0]

    h_matrix = np.concatenate((hankel_u, hankel_x))

    return h_matrix


def create_hankel_pseudo_inverse(h_matrix, dim_u, dim_x):

    # Determine prediction horizon of hankel matrix:
    predict_horizon = int(h_matrix.shape[0]/(dim_u+dim_x)-1)

    # Select relevant rows for pseudo inverse of hankel matrix (used for prediction step)
    u_rows_idx = list(range(0, dim_u*predict_horizon))
    x_rows_idx = list(range(dim_u*(predict_horizon+1),
                      dim_u*(predict_horizon+1)+dim_x))
    relevant_rows = u_rows_idx + x_rows_idx

    h_input_state = h_matrix[relevant_rows, :]
    h_matrix_inv = np.linalg.pinv(h_input_state)

    return h_matrix_inv
