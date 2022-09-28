import numpy as np
from scipy import stats

from .constraint_tightening import ConstraintTightening

# Check behaviour of lti_system class
# Disturbance is turned off during check, so that the behaviour is deterministic


def test_tighten_constraints_on_multivariate_kde():
    """Test if function tighten_constraints_on_multivariate_kde() returns rightly tightened constraints"""

    G_u = np.atleast_2d([1,0])
    g_u = np.atleast_2d([1])
    G_x = np.atleast_2d([1,0,1,0])
    g_x = np.atleast_2d([2])

    size_dataset = 200
    mean = np.zeros([4])
    cov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

    constraint_tightener = ConstraintTightening(G_u,g_u,G_x,g_x,risk_factor=0.95)

    dataset_delta_x = np.random.multivariate_normal(mean, cov, size=size_dataset).transpose()

    kde = stats.gaussian_kde(dataset_delta_x)

    [G_v, g_v, G_z, g_z] = constraint_tightener.tighten_constraints_on_multivariate_kde(kde)

    assert g_z[0] <=2