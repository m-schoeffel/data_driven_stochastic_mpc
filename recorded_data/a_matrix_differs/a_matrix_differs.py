from ctypes import alignment
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# ------------------ Animate dataset ------------------

# The animation is specific to the current dataset and has to be adapted to a new dataset

name_dataset = "a_matrix_differs"
len_traj = 300

def animate_dataset():
    current_wd = os.getcwd()
    path_dataset = os.path.join(current_wd,"recorded_data",name_dataset)

    plt.rc('font', size=20)

    cm = 1/2.54  # centimeters in inches
    my_figsize = (128*cm, 64*cm)

    fig = plt.figure(figsize=my_figsize)
    ax_x_0 = plt.subplot2grid((4,4),(0,0),colspan=4,rowspan=3)
    ax_distr = plt.subplot2grid((4,4),(3,0),colspan=1,rowspan=1)
    ax_weights = plt.subplot2grid((4,4),(3,1),colspan=1,rowspan=1)
    ax_b_coeff = plt.subplot2grid((4,4),(3,2),colspan=1,rowspan=1)
    

    ax_x_0.set_xlabel("Timestep k")
    ax_x_0.set_ylabel("Position x")
    ax_x_0.set_title("Position x with constraints and reference state at timestep k")

    ax_distr.set_xlabel("x")
    ax_distr.set_ylabel("Probability")
    ax_distr.set_title("Estimated and true underlying probability")

    ax_weights.set_xlabel("Sample k-n")
    ax_weights.set_ylabel("Weight")
    ax_weights.set_title("Weights of samples used for KDE")

    ax_b_coeff.set_title("Bhattacharyya coefficient")

    line1, = ax_x_0.plot([], [], color='blue', lw=3,label="Constraints")
    line2, = ax_x_0.plot([], [], color='green', lw=3, ls='--',label="Tightened pseudo constraints")
    line3, = ax_x_0.plot([], [], lw=2,label="Reference trajectory")
    line4 = ax_x_0.scatter([], [], color='grey', lw=2,label="Nominal prediction horizon")
    line5 = ax_x_0.scatter([], [], color='black', lw=2,label="Prediction horizon")
    line6 = ax_x_0.scatter([], [], color='red', lw=2,label="Measured positions x")

    ax_x_0.legend(loc="upper right")

    line_est_pdf, = ax_distr.plot([],[],lw=3, ls='--')
    line_true_pdf, = ax_distr.plot([],[],lw=3)

    line_weights, = ax_weights.plot([],[],color='black',lw=3)

    bar_b_coeff, = ax_b_coeff.bar(1,1)
    bar_b_coeff_text = ax_b_coeff.text(1,.5,'',va="center",ha="center",fontsize=45,color="white")

    timesteps = list(range(0, len_traj))



    fig.tight_layout()

    # Load params from config
    path_config = os.path.join(path_dataset,"config.yaml")
    with open(path_config) as file:
        param = yaml.load(file, Loader=yaml.FullLoader)

    g_x = np.array(param["constraints"]["g_x"])
    x1_constr = np.ones(len_traj)*g_x[0]

    number_eval_points = param["discounted_kde"]["number_eval_points"]
    interv_min = param["discounted_kde"]["interv_min"]
    interv_max = param["discounted_kde"]["interv_max"]

    # Load reference trajectory
    name_ref_traj = param["reference_trajectory"]
    path_ref_traj = os.path.join(path_dataset,name_ref_traj+".csv")
    ref_traj = np.genfromtxt(path_ref_traj, delimiter=',')
    print(ref_traj[:, 25].reshape(-1))

    pdf_interval = np.linspace(interv_min,interv_max,number_eval_points)
    true_pdf = 1/(0.002*np.sqrt(2*np.pi))*np.exp(-0.5*np.power((pdf_interval+0.03)/0.002,2))

    def animate(k):

        path_measured_state = os.path.join(path_dataset,"mpc","measured_states","real_state_k_"+str(k)+".npy")
        measured_state = np.load(path_measured_state)

        # Plot system state x_1 at time k
        ax_x_0.scatter(k, measured_state[0], color='red')


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

        ax_x_0.set_xlim(0, len_traj)
        ax_x_0.set_ylim(-4, 4)

        # Plot distribution
        path_estim_pdf = os.path.join(path_dataset,"disturbance_estimation","disturbance_distribution_x_0","estim_pdf_k_"+str(k)+".npy")
        estimated_pdf = np.load(path_estim_pdf)
        resize_estimated_pdf = estimated_pdf*500
        line_est_pdf.set_data(pdf_interval,resize_estimated_pdf)

        # line_true_pdf.set_data(pdf_interval,true_pdf)
        ax_distr.set_xlim(-0.5,0.5)
        ax_distr.set_ylim(0,12)

        # Plot weights
        path_weights = os.path.join(path_dataset,"disturbance_estimation","weights","weights_k_"+str(k)+".npy")
        weights_x_0 = np.load(path_weights)[0,:]
        idx_weights = list(range(-200,0))
        line_weights.set_data(idx_weights,weights_x_0)
        ax_weights.set_xlim(-200,0)
        ax_weights.set_ylim(0,1)

        # Plot nominal prediction horizon
        path_est_trajectory = os.path.join(path_dataset,"mpc","optimal_trajectories","trajectory_k_"+str(k)+".npy")
        est_traj = np.load(path_est_trajectory)
        nominal_pred_hor_x1 = est_traj[26::4]
        idx_pred_horion = np.atleast_2d(list(range(k+1, k+1+10))).transpose()
        line4.set_offsets(np.hstack([idx_pred_horion,nominal_pred_hor_x1]))

        # Plot prediction horizon
        # Adjust nominal prediction horizon with value of expected disturbance
        exp_dist_x1 = np.dot(estimated_pdf,pdf_interval)
        print(exp_dist_x1)
        pred_hor_x1 = nominal_pred_hor_x1.copy()
        for i in range(1,11):
            pred_hor_x1[(i-1):i] += i*exp_dist_x1
        line5.set_offsets(np.hstack([idx_pred_horion,pred_hor_x1]))

        # Plot Bhattacharyya coefficient
        path_b_coeff = os.path.join(path_dataset,"disturbance_estimation","bhattacharyya_coefficients","b_coeff_k_"+str(k)+".npy")
        b_coeff_x_0 = np.load(path_b_coeff)[0]
        bar_b_coeff.set_height(b_coeff_x_0)
        bar_b_coeff_text.set_text(str(round(b_coeff_x_0,3)))

        # return line1, line2, line3, line4, ax_x_0, line_est_pdf, line_true_pdf, ax_distr, line_weights,bar_b_coeff,bar_b_coeff_text



    anim = animation.FuncAnimation(
        fig, animate, frames=range(0, len_traj), interval=100)

    # path_store_animation = os.path.join(path_dataset,"animation_"+name_dataset+".mp4")
    # writervideo = animation.FFMpegWriter(fps=10)
    # anim.save(path_store_animation, writer=writervideo)

    plt.show()

if __name__ == "__main__":
    animate_dataset()