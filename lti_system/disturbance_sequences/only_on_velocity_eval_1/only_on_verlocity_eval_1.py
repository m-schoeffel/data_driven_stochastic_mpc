import os
import numpy as np

# This script is used to create a disturbance sequence for a 4 state system
# The length of the sequence is 500 samples

# The disturbance on each state is independent in time and not correlated with the disturbances on other states

# There is a gaussian noise on velocity (state x3,x4) with zero mean and variance
# This noise is affecting the position

name_sequence = "only_on_velocity_eval_1"
number_samples = 500


x_3 = np.random.normal(loc=0, scale=0.3,size=number_samples)
x_4 = np.random.normal(loc=0, scale=0.3,size=number_samples)

x_1 = 0.1 * x_3
x_2 = 0.1 * x_4

compl_dist = np.vstack((x_1,x_2,x_3,x_4))

path_dist = os.path.join(os.getcwd(),"lti_system","disturbance_sequences",name_sequence,name_sequence)
np.save(path_dist,compl_dist)
