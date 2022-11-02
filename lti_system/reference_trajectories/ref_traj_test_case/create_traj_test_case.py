#!/usr/bin/python3
import os
import numpy as np

name_traj = "ref_traj_test_case"

# reference trajectory for 4 state system
# used to test if state constraint is held with probability specified in risk parameter
# hold 2.0 with x1 for 2000 samples
# (sampling rate is generally assumed to be 10Hz)
# States 2-4 supposed to stay 0 for the whole duration
traj = np.array([[2.0],[0],[0],[0]])
traj = np.tile(traj,2000)

traj = np.asarray(traj)
traj_path = os.path.join("lti_system","reference_trajectories",name_traj,name_traj+".csv")
np.savetxt(traj_path,traj,delimiter=",")
