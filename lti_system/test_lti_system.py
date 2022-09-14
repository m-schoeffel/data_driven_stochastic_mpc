import numpy as np

from .lti_system import LTISystem
from .disturbance import Disturbance

# Check behaviour of lti_system class
# Disturbance is turned off during check, so that the behaviour is deterministic


def test_2state_lti_system():
    """Check behaviour of lti system with 2 states and 1 input"""

    x = np.ones([2, 1])
    A = np.array([[1, 1], [0, 1]])
    B = np.array([[0], [1]])

    disturbance = Disturbance(["gaussian"])

    lti_system = LTISystem(x, A, B, disturbance)

    next_x = lti_system.next_step(u=1, add_disturbance=False)

    assert next_x[0] == 2
    assert next_x[1] == 2

    next_x = lti_system.next_step(u=-5, add_disturbance=False)

    assert next_x[0] == 4
    assert next_x[1] == -3


def test_4state_lti_system():
    """Check behaviour of lti system with 4 states and 2 input"""

    x = np.array([[1], [2], [0], [5]])
    A = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
    B = np.array([[0, 0], [1, 0], [0, 0], [0, 1]])

    disturbance = Disturbance(["gaussian", "uniform", "gaussian", "gaussian"])

    lti_system = LTISystem(x, A, B, disturbance)

    next_u = np.array([[5], [3]])
    next_x = lti_system.next_step(u=next_u, add_disturbance=False)

    assert next_x[0] == 3
    assert next_x[1] == 7
    assert next_x[2] == 5
    assert next_x[3] == 8

    next_u = np.array([[-8], [3]])
    next_x = lti_system.next_step(u=next_u, add_disturbance=False)

    assert next_x[0] == 10
    assert next_x[1] == -1
    assert next_x[2] == 13
    assert next_x[3] == 11
