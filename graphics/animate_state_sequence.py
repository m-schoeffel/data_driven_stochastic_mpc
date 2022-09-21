import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from config import load_parameters

# ------------------ Animate state sequence ------------------

# The animation is specific to the current system and has to be adapted to a new system


def animate_state_sequence(state_storage, g_z_storage, ref_traj, pred_hor_storage):

    cm = 1/2.54  # centimeters in inches
    my_figsize = (128*cm, 64*cm)

    fig = plt.figure(figsize=my_figsize)
    ax = plt.axes(xlim=(-4, 4), ylim=(-4, 4))

    line1, = ax.plot([], [], color='blue', lw=3)
    line2, = ax.plot([], [], color='green', lw=3, ls='--')
    line3, = ax.plot([], [], lw=2)
    line4 = ax.scatter([], [], color='black', lw=2)

    len_traj = state_storage.shape[1]
    timesteps = list(range(0, len_traj))

    # prepare plots for joint constraints for x1 and x2
    # Todo: Make plottet constraints truly flexible
    constraints = load_parameters.load_constraints()

    G_x = np.array(constraints["G_x"])
    g_x = np.array(constraints["g_x"])
    x1_constr = np.ones(len_traj)*g_x[0]

    def animate(k):
        # Plot system state x_1 at time k
        ax.scatter(k, state_storage[0, k], color='red')

        # Plot real constraint
        line1.set_data(timesteps, x1_constr)

        # Plot tightened pseudo constraints
        curr_tight_x1_constr = g_z_storage[k][0]
        print(f"curr_tight_x1_constr:\n{curr_tight_x1_constr}")
        curr_tight_x1_constr_array = np.ones(len_traj)*curr_tight_x1_constr
        line2.set_data(timesteps, curr_tight_x1_constr_array)

        # Plot reference trajectory
        line3.set_data(timesteps, ref_traj[0, 0:len_traj])

        # Plot prediction horizon
        idx_pred_horion = np.atleast_2d(list(range(k+1, k+1+10))).transpose()
        pred_hor = np.hstack([idx_pred_horion, pred_hor_storage[k+1][0::4]])
        line4.set_offsets(pred_hor)

        ax.set_xlim(0, len_traj)

    anim = animation.FuncAnimation(
        fig, animate, frames=range(0, state_storage.shape[1]), interval=300)

    # f = r"first_reverence_tracking.mp4"
    # writervideo = animation.FFMpegWriter(fps=10)
    # anim.save(f, writer=writervideo)

    plt.show()
