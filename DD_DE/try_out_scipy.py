import numpy as np

from scipy.optimize import minimize, LinearConstraint,NonlinearConstraint


def constr_cube(x):
    A=np.array([[0,1,0],[0,0,1],[1,0,0]])
    b=A@x
    return b[0]*b[0]+b[1]*b[1]+b[2]*b[2]

# constraint = LinearConstraint(np.ones(n_buyers), lb=n_shares, ub=n_shares)
constraint = NonlinearConstraint(constr_cube, lb=1, ub=np.inf)

A = np.array([1,0,0])
B = np.array([0,1,0])

constraint_2 = LinearConstraint(A, lb=0, ub=0)
constraint_3 = LinearConstraint(B, lb=0, ub=0)



def objective_function(x):

    return x[0]+x[1]+x[2]

res = minimize(

    objective_function,

    np.array([2,2,2]),

    constraints=[constraint_2,constraint_3,constraint],

)

print(res)