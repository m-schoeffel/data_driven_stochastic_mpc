import numpy as np

from . import hankel_helpers


def test_create_hankel_matrix():
    """Test function create_hankel_matrix"""

    # Test if a hankel matrix is correctly built
    # The input and state sequences do not correspond to a LTI-System

    input_sequence = np.atleast_2d([1, 2, 3, 4, 5, 6])
    state_sequence = np.atleast_2d([4, 5, 6, 7, 8, 9])
    predic_hori_size = 3

    h_matrix = hankel_helpers.create_hankel_matrix(
        input_sequence, state_sequence, predic_hori_size)

    h_matrix_comp = np.atleast_2d([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [
                                  4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9]])

    assert (h_matrix == h_matrix_comp).all()
