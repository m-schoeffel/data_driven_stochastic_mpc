from this import d
import numpy as np

from . import _data_driven_mpc

# Check behaviour of data_driven_mpc with simple test cases


def test_get_new_u():
    """Test function get_new_u"""
    # Underlying system is x_(k+1)=x_k+u_k

    input_sequence = np.atleast_2d([1, -2, 0, 2, -4, 5])
    state_sequence = np.atleast_2d([3, 4, 2, 2, 4, 0])

    predic_hori_size = 1

    state_cost = np.atleast_2d(50)
    input_cost = np.atleast_2d(0)

    # State tightened constraints
    G_v = np.atleast_2d(1.0)
    g_v = np.atleast_2d(4.0)

    G_z = np.atleast_2d(1.0)
    g_z = np.atleast_2d(6.0)

    dd_mpc = _data_driven_mpc.DataDrivenMPC(
        input_sequence, state_sequence, predic_hori_size, state_cost, input_cost)

    current_x = np.atleast_2d([1])
    [next_u, x_pred, prediction_horizon] = dd_mpc.get_new_u(
        current_x, G_v, g_v, G_z, g_z)

    assert np.abs(next_u-(-1)) <= 0.0001
    assert np.abs(x_pred-0) <= 0.0001
