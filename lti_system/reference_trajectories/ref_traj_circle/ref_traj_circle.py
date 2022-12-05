#!/usr/bin/python3
import os
import numpy as np

name_traj = "ref_traj_circle"

# reference trajectory for 4 state system

# (sampling rate is generally assumed to be 10Hz)
x_traj = np.array([np.sin(2*np.pi*x/50) for x in range(0,500)])
y_traj = np.array([np.cos(2*np.pi*x/50) for x in range(0,500)])

v_x_y = np.zeros([2,500])

traj = np.vstack([x_traj,y_traj,v_x_y])
traj_path = os.path.join("lti_system","reference_trajectories",name_traj,name_traj+".csv")
np.savetxt(traj_path,traj,delimiter=",")
