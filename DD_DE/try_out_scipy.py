import numpy as np

from scipy.optimize import minimize, LinearConstraint


n_buyers = 10

n_shares = 15


np.random.seed(10)

prices = np.random.random(n_buyers)

money_available = np.random.randint(1, 4, n_buyers)

n_shares_per_buyer = money_available / prices

print(prices, money_available, n_shares_per_buyer, sep="\n")

for_constraint = np.ones((2,n_buyers))
fc_2=n_shares*np.ones(2)
print(for_constraint.shape,fc_2.shape,sep="\n------------------------\n")

constraint = LinearConstraint(np.ones(n_buyers), lb=n_shares, ub=n_shares)

bounds = [(0, n) for n in n_shares_per_buyer]

def objective_function(x, prices):

    return -x.dot(prices)+sum(x**-2)

res = minimize(

    objective_function,

    x0=10 * np.random.random(n_buyers),

    args=(prices,),

    constraints=constraint,

    bounds=bounds,

)

print(res)