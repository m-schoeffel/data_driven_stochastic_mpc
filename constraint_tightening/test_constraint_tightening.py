import numpy as np
from scipy import stats

# from constraint_tightening.constraint_tightening import ConstraintTightening
from constraint_tightening import _constraint_tightening


# Check behaviour of lti_system class
# Disturbance is turned off during check, so that the behaviour is deterministic


def test_tighten_constraints_on_multivariate_kde_one_state_constraint():
    """Test function tighten_constraints_on_multivariate_kde() with a positive constraint, which only affects a single state
        Constraint: X_1<=2
        Disturbance: Multivariate gaussian distribution (independent dimensions) with mean 0 and variance 2 in all directions
        Expected outcome is g[0]=-0.8
    """

    G_u = np.atleast_2d([1,0])
    g_u = np.atleast_2d([1])
    G_x = np.atleast_2d([1,0,0,0])
    g_x = np.atleast_2d([2])

    size_dataset = 200
    mean = np.zeros([4])
    cov = np.array([[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,2]])

    constraint_tightener = _constraint_tightening.ConstraintTightening(G_u,g_u,G_x,g_x,risk_factor=0.975)

    dataset_delta_x = np.random.multivariate_normal(mean, cov, size=size_dataset).transpose()

    kde = stats.gaussian_kde(dataset_delta_x)

    [G_v, g_v, G_z, g_z] = constraint_tightener.tighten_constraints_on_multivariate_kde(kde)

    assert -1.4 <= g_z[0] <=-0.7

def test_tighten_constraints_on_multivariate_kde_joint_constraint():
    """Test function tighten_constraints_on_multivariate_kde() with a joint constraint
        Constraint: 2X_1 - X_3 <= 2
        Disturbance: Multivariate gaussian distribution (independent dimensions) with mean 0 and variance 2 in all directions
        Expected outcome is g[0]=-4.32
    """

    G_u = np.atleast_2d([1,0])
    g_u = np.atleast_2d([1])
    G_x = np.atleast_2d([2,0,-1,0])
    g_x = np.atleast_2d([2])

    size_dataset = 200
    mean = np.zeros([4])
    cov = np.array([[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,2]])

    constraint_tightener = _constraint_tightening.ConstraintTightening(G_u,g_u,G_x,g_x,risk_factor=0.975)

    dataset_delta_x = np.random.multivariate_normal(mean, cov, size=size_dataset).transpose()

    kde = stats.gaussian_kde(dataset_delta_x)

    [G_v, g_v, G_z, g_z] = constraint_tightener.tighten_constraints_on_multivariate_kde(kde)

    assert -4.9 <= g_z[0] <=-3.5



if __name__ == "__main__":
    test_tighten_constraints_on_multivariate_kde_joint_constraint()