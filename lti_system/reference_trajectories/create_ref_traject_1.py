#!/usr/bin/python3

import numpy as np

# reference trajectory for 4 state system
# alternate 0.1 and 1.9 for x-coordinate (state 1) in 2s intervals 
# (sampling rate is gerally assumed to be 10Hz)
# States 2-4 supposed to stay 0 for the whole duration
traj_a = np.array([[0.1],[0],[0],[0]])
traj_a = np.tile(traj_a,20)

traj_b = np.array([[1.9],[0],[0],[0]])
traj_b = np.tile(traj_b,20)

# Total running time is 60s
traj = np.hstack([traj_a,traj_b])
traj = np.tile(traj,15)

traj = np.asarray(traj)
np.savetxt("lti_system/reference_trajectories/ref_traj_1.csv",traj,delimiter=",")
