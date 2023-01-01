from ctypes import alignment
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.font_manager as font_manager

# ------------------ Animate dataset ------------------

# The animation is specific to the current dataset and has to be adapted to a new dataset

name_dataset = "eval_2_sinus"
len_traj = 450

current_wd = os.getcwd()
path_dataset = os.path.join(current_wd,"recorded_data",name_dataset)

plt.rc('font', size=8)
# plt.rc('text', usetex=True)
plt.rc('font', family='serif')

without_serif= font_manager.FontProperties(family='sans-serif',
                                   style='normal', size=7.5)

cm = 1/2.54  # centimeters in inches
my_figsize = (15*cm, 8*cm)

fig = plt.figure(figsize=my_figsize)
fig.tight_layout()

plt_state_dev = plt.subplot2grid((2,1),(0,0),colspan=1,rowspan=1)
plt_bc = plt.subplot2grid((2,1),(1,0),colspan=1,rowspan=1)

plt_state_dev.set_ylabel("$r_{k}-x_{\mathrm{m},k}$")
plt_state_dev.set_xlabel("Timestep $k$")

plt_bc.set_ylabel("Bhatt. coeff. $b_c$")
plt_bc.set_xlabel("Timestep $k$")

# Load all measured x values and Bhattacharrya coefficients
x_measured = np.zeros(len_traj)
b_coeff = np.zeros(len_traj)
for k in range(0,len_traj):
        path_measured_state = os.path.join(path_dataset,"mpc","measured_states","real_state_k_"+str(k)+".npy")
        x_measured[k] = np.load(path_measured_state)[0]

        path_b_coeff = os.path.join(path_dataset,"disturbance_estimation","bhattacharyya_coefficients","b_coeff_k_"+str(k)+".npy")
        b_coeff[k] = np.load(path_b_coeff)[0]

# Load reference trajectory
path_ref_traj = os.path.join(path_dataset,"ref_traj_sinus.csv")
ref_traj = np.genfromtxt(path_ref_traj, delimiter=',')[0,0:450]

k = list(range(0,len_traj))

plt_state_dev.plot(k,ref_traj-x_measured)
plt_bc.plot(k,b_coeff)

fig.tight_layout()
path_cur_plot = os.path.join(path_dataset,name_dataset+"_plot_deviations_b_c.pdf")
fig.savefig(path_cur_plot, format="pdf", bbox_inches="tight")

plt.show()
