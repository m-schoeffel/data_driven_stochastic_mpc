import os
import numpy as np

# This script is used to create a disturbance sequence for a 4 state system
# The length of the sequence is 2000 samples
# Used to test if state constraint is held with probability specified in risk parameter
# -> Disturbance on x1 should match disturbance specified by 
# The disturbance on each state is independent in time and not correlated with the disturbances on other states

# The underlying distribution of the disturbances on states 2-4 is a gaussian distribution with mean 0 and standard deviation 0.02 (reduce effect, but no disturbance leads to numeric problems in trying to estimate disturbance)

# The underlying distribution of the disturbances on x1 should match how the disturbance estimator is initialized

name_sequence = "gaussian_test_case"
number_samples = 2000

x_1 = np.random.normal(loc=0, scale=0.15,size=number_samples)

x_2 = np.random.normal(loc=0, scale=0.02,size=number_samples)
x_3 = np.random.normal(loc=0, scale=0.02,size=number_samples)
x_4 = np.random.normal(loc=0, scale=0.02,size=number_samples)

compl_dist = np.vstack((x_1,x_2,x_3,x_4))

path_dist = os.path.join(os.getcwd(),"lti_system","disturbance_sequences",name_sequence,name_sequence)
np.save(path_dist,compl_dist)
