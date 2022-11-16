#!/usr/bin/python3
import os
import numpy as np

name_traj = "ref_sinus_eval"
length = 500

# reference trajectory for 4 state system
# Length is 500 samples
# Reference for x1 is a sinus curve
# Reference for x2-4 is 0

traj_x1 = np.array([np.sin(np.pi*k/25)+1 for k in range(0,length)])

traj_x2_3_4 = np.zeros([3,length])

# Total running time is 60s
traj = np.vstack([traj_x1,traj_x2_3_4])

traj_path = os.path.join("lti_system","reference_trajectories",name_traj,name_traj+".csv")
np.savetxt(traj_path,traj,delimiter=",")
