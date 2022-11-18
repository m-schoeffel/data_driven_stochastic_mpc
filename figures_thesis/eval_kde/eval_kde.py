import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

# Evaluate the Kernel Density Estimation
# Record the mean square error between the estimated distriubtion on the real distribution 
# dependending on different sample sizes over 1000 runs.

# The number of samples used ranges from 1 to 1000

mean=0
std_dev=1

x_eval = np.linspace(-4, 4, 4001)

real_distr = 1/(std_dev*np.sqrt(2*np.pi))*np.exp(-0.5*np.power((x_eval-mean)/std_dev,2))

# test_s = np.random.normal(loc=mean, scale=std_dev,size=1000)
# kde = stats.gaussian_kde(test_s)

# test_d = kde.evaluate(x_eval)

# print(np.sum(np.abs(test_d-real_distr)))

# plt.plot(x_eval,test_d,color='r')
# plt.plot(x_eval,real_distr,color='b')
# plt.show()

max_num_samp = 1000
num_runs = 100

square_error_storage = list()

for num in range(2,max_num_samp+1):
    print(num)
    mean_error = 0
    for run in range(0,num_runs):
        samples = np.random.normal(loc=mean, scale=std_dev,size=num)

        kde = stats.gaussian_kde(samples)

        prob_distribution = kde.evaluate(x_eval)

        error = np.sum(np.abs(prob_distribution-real_distr))
        mean_error += np.power(error,2)/num_runs
    square_error_storage.append(mean_error)

x_plot = list(range(1,1001))
plt.plot(square_error_storage)
plt.show()


save_error = np.array(square_error_storage)
np.save("mean_square_error",save_error)