#!/usr/bin/python3
import os
import numpy as np

name_traj = "ref_traj_oval"

# reference trajectory for 4 state system

# (sampling rate is generally assumed to be 10Hz)
x_traj_1 = np.atleast_2d([np.sin(2*np.pi*x/200) for x in range(0,200)])
y_traj_1 = np.atleast_2d([1.3-1.3*np.cos(2*np.pi*x/200) for x in range(0,200)])

x_traj_2 = np.atleast_2d([x/10 for x in range(0,50)])
y_traj_2 = np.zeros([1,50])

x_traj_3 = np.atleast_2d([np.sin(2*np.pi*x/200)+5 for x in range(0,250)])
y_traj_3 = np.atleast_2d([1.3-1.3*np.cos(2*np.pi*x/200) for x in range(0,250)])

x_traj = np.hstack([x_traj_1,x_traj_2,x_traj_3])
y_traj = np.hstack([y_traj_1,y_traj_2,y_traj_3])

v_x_y = np.zeros([2,500])

traj = np.vstack([x_traj,y_traj,v_x_y])
traj_path = os.path.join("lti_system","reference_trajectories",name_traj,name_traj+".csv")
np.savetxt(traj_path,traj,delimiter=",")
