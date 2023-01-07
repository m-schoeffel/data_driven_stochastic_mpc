from ctypes import alignment
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.font_manager as font_manager

# ------------------ Animate dataset ------------------

# The animation is specific to the current dataset and has to be adapted to a new dataset

name_dataset = "eval_7_const_1_1"
len_traj = 450

true_mean = 0
true_std_dev = 0.1

def animate_dataset():
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

    ax_x_0 = plt.subplot2grid((2,3),(0,0),colspan=3,rowspan=1)
    ax_distr = plt.subplot2grid((2,3),(1,0),colspan=1,rowspan=1)
    ax_weights = plt.subplot2grid((2,3),(1,1),colspan=1,rowspan=1)
    ax_b_coeff = plt.subplot2grid((2,3),(1,2),colspan=1,rowspan=1)
    
    ax_x_0.set_xlabel(r"Timestep $k$",usetex=True)
    ax_x_0.set_ylabel(r"Position $x$",usetex=True)
    # ax_x_0.set_title("Position x with constraints and reference state at timestep k")

    ax_distr.set_xlabel(r"Disturbance $\Delta x$",usetex=True)
    ax_distr.set_ylabel(r"Probability $f(\Delta x)$",usetex=True)
    # ax_distr.set_title("Estimated and true underlying probability")

    ax_weights.set_xlabel(r"Index $i$",usetex=True)
    ax_weights.set_ylabel(r"Weight $w$",usetex=True)
    # ax_weights.set_title("Weights of samples used for KDE")

    ax_b_coeff.set_xlabel(r"Bhattacharyya coeff. $b_c$",usetex=True)
    # ax_b_coeff.set_title("Bhattacharyya coefficient")

    line1, = ax_x_0.plot([], [], color='orange', lw=0.8)#,label="Constraints")
    line2, = ax_x_0.plot([], [], color='green', lw=0.8, ls='--')#,label="Tightened pseudo constraints")
    line3, = ax_x_0.plot([], [], lw=0.8,label="Reference trajectory")
    line4 = ax_x_0.scatter([], [], color='grey', s=0.1,label="Nominal prediction horizon")
    line5 = ax_x_0.scatter([], [], color='black', s=0.2,label="Prediction horizon")
    line6 = ax_x_0.scatter([], [], color='red', s=0.1,label="Measured positions $x$")


    # ax_x_0.legend(loc="right",prop=without_serif)

    line_est_pdf, = ax_distr.plot([],[],lw=0.8, ls='--',label=r"$f_\mathrm{est}(\Delta x)$")
    # line_true_pdf, = ax_distr.plot([],[],lw=0.8,label=r"$f_\mathrm{true}(\Delta x)$")
    ax_distr.legend(prop=without_serif)

    line_weights, = ax_weights.plot([],[],color='black',lw=0.8,label=r"Weights $w_i$")
    ax_weights.legend(prop=without_serif)

    bar_b_coeff, = ax_b_coeff.bar(1,1,label=r"$b_c$")
    ax_b_coeff.xaxis.set_ticklabels([])
    bar_b_coeff_text = ax_b_coeff.text(1,.5,'',va="center",ha="center",fontsize=8,color="black")
    ax_b_coeff.legend(prop=without_serif)

    timesteps = list(range(0, len_traj))



    fig.tight_layout()

    # Load params from config
    path_config = os.path.join(path_dataset,"config.yaml")
    with open(path_config) as file:
        param = yaml.load(file, Loader=yaml.FullLoader)

    g_x = np.array(param["constraints"]["g_x"])
    x1_constr = np.ones(len_traj)*g_x[0]

    path_plots = os.path.join(path_dataset, "eval_plots_no_legend")
    dir_exists = os.path.exists(path_plots)
    if not dir_exists:
        os.mkdir(path_plots)

    number_eval_points = param["discounted_kde"]["number_eval_points"]
    interv_min = param["discounted_kde"]["interv_min"]
    interv_max = param["discounted_kde"]["interv_max"]

    # Load reference trajectory
    name_ref_traj = param["reference_trajectory"]
    path_ref_traj = os.path.join(path_dataset,name_ref_traj+".csv")
    ref_traj = np.genfromtxt(path_ref_traj, delimiter=',')
    print(ref_traj[:, 25].reshape(-1))

    pdf_interval = np.linspace(interv_min,interv_max,number_eval_points)
    true_pdf = 1/(true_std_dev*np.sqrt(2*np.pi))*np.exp(-0.5*np.power((pdf_interval-true_mean)/true_std_dev,2))

    def animate(k):

        path_measured_state = os.path.join(path_dataset,"mpc","measured_states","real_state_k_"+str(k)+".npy")
        measured_state = np.load(path_measured_state)

        # Plot system state x_1 at time k
        ax_x_0.scatter(k, measured_state[0], s=0.1,color='red')


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
        ax_x_0.set_ylim(1.2, 2.8)

        # Plot distribution
        path_estim_pdf = os.path.join(path_dataset,"disturbance_estimation","disturbance_distribution_x_0","estim_pdf_k_"+str(k)+".npy")
        estimated_pdf = np.load(path_estim_pdf)
        resize_estimated_pdf = estimated_pdf*number_eval_points/(interv_max-interv_min)
        line_est_pdf.set_data(pdf_interval,resize_estimated_pdf)

        # line_true_pdf.set_data(pdf_interval,true_pdf)
        ax_distr.set_xlim(-0.4,0.4)
        ax_distr.set_ylim(0,np.max(resize_estimated_pdf)+1)

        # Plot weights
        path_weights = os.path.join(path_dataset,"disturbance_estimation","weights","weights_k_"+str(k)+".npy")
        weights_x_0 = np.load(path_weights)[0,:]
        weights_x_0 = weights_x_0/np.sum(weights_x_0) # Normalize weights
        idx_weights = list(range(-200,0))
        line_weights.set_data(idx_weights,weights_x_0)
        ax_weights.set_xlim(-200,0)
        ax_weights.set_ylim(0,weights_x_0[-1])
        ax_weights.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

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

        if k==0 or k==50 or k==100 or k==150 or k==200 or k==250 or k==300 or k==399 or k==400 or k==449 or k==499:
            path_cur_plot = os.path.join(path_plots,name_dataset+"_k_"+str(k)+".pdf")
            fig.savefig(path_cur_plot, format="pdf")#, bbox_inches="tight")


        # return line1, line2, line3, line4, ax_x_0, line_est_pdf, line_true_pdf, ax_distr, line_weights,bar_b_coeff,bar_b_coeff_text



    anim = animation.FuncAnimation(
        fig, animate, frames=range(0, len_traj), interval=100,repeat=False)

    # path_store_animation = os.path.join(path_dataset,"animation_"+name_dataset+".mp4")
    # writervideo = animation.FFMpegWriter(fps=10)
    # anim.save(path_store_animation, writer=writervideo)

    plt.show()

if __name__ == "__main__":
    animate_dataset()