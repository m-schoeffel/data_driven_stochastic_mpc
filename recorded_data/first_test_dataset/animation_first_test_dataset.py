import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from config import _load_parameters

# ------------------ Animate dataset ------------------

# The animation is specific to the dataset and has to be adapted to a new dataset

def animate_dataset():
    current_wd = os.getcwd()
    path_dataset = os.path.join(current_wd,"recorded_data","first_test_dataset")

    plt.rc('font', size=20)

    cm = 1/2.54  # centimeters in inches
    my_figsize = (128*cm, 64*cm)

    fig = plt.figure(figsize=my_figsize)
    ax = plt.axes(xlim=(0, 200), ylim=(-4, 4))

    ax.set_xlabel("Timestep k")
    ax.set_ylabel("Position x")
    ax.set_title("Position x with constraints and reference state at timestep k")

    line1, = ax.plot([], [], color='blue', lw=3)
    line2, = ax.plot([], [], color='green', lw=3, ls='--')
    line3, = ax.plot([], [], lw=2)
    line4 = ax.scatter([], [], color='black', lw=2)

    len_traj = 200
    timesteps = list(range(0, len_traj))

    # Load reference trajectory
    path_ref_traj = os.path.join(path_dataset,"ref_traj_1.csv")
    ref_traj = np.genfromtxt(path_ref_traj, delimiter=',')
    print(ref_traj[:, 25].reshape(-1))

    # Load constraints from config
    path_config = os.path.join(path_dataset,"config.yaml")
    with open(path_config) as file:
        param = yaml.load(file, Loader=yaml.FullLoader)

    g_x = np.array(param["constraints"]["g_x"])
    x1_constr = np.ones(len_traj)*g_x[0]

    def animate(k):

        path_measured_state = os.path.join(path_dataset,"mpc","measured_states","real_state_k_"+str(k)+".npy")
        measured_state = np.load(path_measured_state)

        # Plot system state x_1 at time k
        ax.scatter(k, measured_state[0], color='red')

        # Plot real constraint
        line1.set_data(timesteps, x1_constr)

        # Plot tightened pseudo constraints
        path_g_z = os.path.join(path_dataset,"constraint_tightening","tightened_constraints","g_z_k_"+str(k)+".npy")
        curr_tight_x1_constr = np.load(path_g_z)[0]
        print(f"curr_tight_x1_constr:\n{curr_tight_x1_constr}")
        curr_tight_x1_constr_array = np.ones(len_traj)*curr_tight_x1_constr
        line2.set_data(timesteps, curr_tight_x1_constr_array)

        # Plot reference trajectory
        line3.set_data(timesteps, ref_traj[0, 0:len_traj])

        # Plot prediction horizon
        path_est_trajectory = os.path.join(path_dataset,"mpc","optimal_trajectories","trajectory_k_"+str(k)+".npy")
        est_traj = np.load(path_est_trajectory)
        pred_hor_x1 = est_traj[26::4]
        idx_pred_horion = np.atleast_2d(list(range(k+1, k+1+10))).transpose()
        line4.set_offsets(np.hstack([idx_pred_horion,pred_hor_x1]))

        ax.set_xlim(0, len_traj)

    anim = animation.FuncAnimation(
        fig, animate, frames=range(0, 200), interval=300)

    # f = r"first_reverence_tracking.mp4"
    # writervideo = animation.FFMpegWriter(fps=10)
    # anim.save(f, writer=writervideo)

    plt.show()

if __name__ == "__main__":
    animate_dataset()