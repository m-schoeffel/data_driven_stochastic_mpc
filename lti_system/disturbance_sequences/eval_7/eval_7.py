import os
import numpy as np

# This script is used to create a disturbance sequence for a 4 state system
# The length of the sequence is 500 samples
# The disturbance on each state is independent in time and not correlated with the disturbances on other states

# This disturbance is used to simulate "no disturbance" aka minimal measurement noise during execution

name_sequence = "eval_7"

x_1 = np.random.normal(loc=0, scale=0.005,size=500)
x_2 = np.random.normal(loc=0, scale=0.005,size=500)
x_3 = np.random.normal(loc=0, scale=0.005,size=500)
x_4 = np.random.normal(loc=0, scale=0.005,size=500)

compl_dist = np.vstack((x_1,x_2,x_3,x_4))

path_dist = os.path.join(os.getcwd(),"lti_system","disturbance_sequences",name_sequence,name_sequence)
np.save(path_dist,compl_dist)
