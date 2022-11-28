import os
import numpy as np

# This script is used to create a disturbance sequence for a 4 state system
# The length of the sequence is 500 samples
# The disturbance on each state is zero

# This disturbance is used to simulate no disturbance

name_sequence = "no_noise"

compl_dist = np.zeros([4,500])

path_dist = os.path.join(os.getcwd(),"lti_system","disturbance_sequences",name_sequence,name_sequence)
np.save(path_dist,compl_dist)
