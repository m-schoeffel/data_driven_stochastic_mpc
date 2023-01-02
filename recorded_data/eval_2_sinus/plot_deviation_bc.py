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
my_figsize = (15*cm, 4*cm)






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

# Plot state deviations
fig = plt.figure(1,figsize=my_figsize)
fig.tight_layout()

plt.ylabel("${x}_{\mathrm{m},k}-{x}_{\mathrm{ref},k}$",usetex=True)
plt.xlabel("Timestep $k$",usetex=True)

plt.plot(k,-(ref_traj-x_measured),lw=0.8)

plt.tight_layout()
path_cur_plot = os.path.join(path_dataset,name_dataset+"_plot_deviations.pdf")
plt.savefig(path_cur_plot, format="pdf", bbox_inches="tight")

# Plot Bhatt. coeff.
fig = plt.figure(2,figsize=my_figsize)
fig.tight_layout()

plt.ylabel("Bhatt. coeff. $b_c$",usetex=True)
plt.xlabel("Timestep $k$",usetex=True)

plt.plot(k,b_coeff,lw=0.8)

plt.tight_layout()
path_cur_plot = os.path.join(path_dataset,name_dataset+"_b_c.pdf")
plt.savefig(path_cur_plot, format="pdf", bbox_inches="tight")

plt.show()
