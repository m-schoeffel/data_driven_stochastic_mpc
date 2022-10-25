import os
import numpy as np

# This script is used to create a disturbance sequence for a 4 state system
# The length of the sequence is 500 samples
# The disturbance on each state is independent in time and not correlated with the disturbances on other states

# The underlying distribution of the disturbances on states 2-4 is a gaussian distribution with mean 0 and standard deviation 0.1

# The underlying distribution of the disturbances on x1 changes at sample 250
# The distribution for the first 250 samples is a gaussian distribution with mean -0.15 and standard deviation 0.15
# The distribution for the last 250 samples is a gaussian distribution with mean 0.1 and standard deviation 0.1

name_sequence = "gaussian_gaussian_500"

x_1_first_250 = np.random.normal(loc=-0.15, scale=0.15,size=250)
x_1_last_250 = np.random.normal(loc=0.1, scale=0.1,size=250)
x_1 = np.hstack((x_1_first_250,x_1_last_250))

x_2 = np.random.normal(loc=0, scale=0.1,size=500)
x_3 = np.random.normal(loc=0, scale=0.1,size=500)
x_4 = np.random.normal(loc=0, scale=0.1,size=500)

compl_dist = np.vstack((x_1,x_2,x_3,x_4))

path_dist = os.path.join(os.getcwd(),"lti_system","disturbance_sequences",name_sequence,name_sequence)
np.save(path_dist,compl_dist)
