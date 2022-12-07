from ctypes import alignment
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.font_manager as font_manager

# ------------------ Animate dataset ------------------

# The animation is specific to the current dataset and has to be adapted to a new dataset

name_dataset = "eval_5_oval"
len_traj = 450

true_mean_x = 0.1
true_std_dev_x = 0.05

true_mean_y = 0.15
true_std_dev_y = 0.01


def animate_dataset():
    current_wd = os.getcwd()
    path_dataset = os.path.join(current_wd,"recorded_data",name_dataset)

    plt.rc('font', size=8)
    # plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    without_serif= font_manager.FontProperties(family='sans-serif',
                                   style='normal', size=7.5)

    cm = 1/2.54  # centimeters in inches
    my_figsize = (15*cm, 15*cm)

    fig = plt.figure(figsize=my_figsize)
    fig.tight_layout()

    ax_x_0 = plt.subplot2grid((3,2),(0,0),colspan=2,rowspan=2)
    ax_distr_x = plt.subplot2grid((3,2),(2,0),colspan=1,rowspan=1)
    ax_distr_y = plt.subplot2grid((3,2),(2,1),colspan=1,rowspan=1)
    
    ax_x_0.set_xlabel(r"Position $x$",usetex=True)
    ax_x_0.set_ylabel(r"Position $y$",usetex=True)
    # ax_x_0.set_title("Position x with constraints and reference state at timestep k")

    ax_distr_x.set_xlabel(r"Disturbance $\Delta x$",usetex=True)
    ax_distr_x.set_ylabel(r"Probability $f(\Delta x)$",usetex=True)
    # ax_distr_x.set_title("Estimated and true underlying probability")

    ax_distr_y.set_xlabel(r"Disturbance $\Delta y$",usetex=True)
    ax_distr_y.set_ylabel(r"Probability $f(\Delta y)$",usetex=True)



    line1, = ax_x_0.plot([], [], color='orange', lw=0.8,label="Constraints")
    line2, = ax_x_0.plot([], [], color='green', lw=0.8, ls='--',label="Tightened pseudo constraints")
    line3, = ax_x_0.plot([], [], lw=0.8,label="Reference trajectory")
    line4 = ax_x_0.scatter([], [], color='grey', s=0.1,label="Nominal prediction horizon")
    line5 = ax_x_0.scatter([], [], color='black', s=0.2,label="Prediction horizon")
    line6 = ax_x_0.scatter([], [], color='red', s=0.1,label="Measured positions $x$")


    ax_x_0.legend(loc="upper right",prop=without_serif)

    line_est_pdf_x, = ax_distr_x.plot([],[],lw=0.8, ls='--',label=r"$f_\mathrm{est}(\Delta x)$")
    line_true_pdf_x, = ax_distr_x.plot([],[],lw=0.8,label=r"$f_\mathrm{true}(\Delta x)$")
    ax_distr_x.legend(prop=without_serif)

    line_est_pdf_y, = ax_distr_y.plot([],[],lw=0.8, ls='--',label=r"$f_\mathrm{est}(\Delta y)$")
    line_true_pdf_y, = ax_distr_y.plot([],[],lw=0.8,label=r"$f_\mathrm{true}(\Delta y)$")
    ax_distr_y.legend(prop=without_serif)



    timesteps = list(range(0, len_traj))



    fig.tight_layout()

    # Load params from config
    path_config = os.path.join(path_dataset,"config.yaml")
    with open(path_config) as file:
        param = yaml.load(file, Loader=yaml.FullLoader)

    g_x = np.array(param["constraints"]["g_x"])
    first_constr = np.ones(len_traj)*g_x[0]

    path_plots = os.path.join(path_dataset, "eval_plots")
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
    true_pdf_x = 1/(true_std_dev_x*np.sqrt(2*np.pi))*np.exp(-0.5*np.power((pdf_interval-true_mean_x)/true_std_dev_x,2))
    true_pdf_y = 1/(true_std_dev_y*np.sqrt(2*np.pi))*np.exp(-0.5*np.power((pdf_interval-true_mean_y)/true_std_dev_y,2))

    def animate(k):

        path_measured_state = os.path.join(path_dataset,"mpc","measured_states","real_state_k_"+str(k)+".npy")
        measured_state = np.load(path_measured_state)

        # Plot system state x|y at current timestep k
        ax_x_0.scatter(measured_state[0], measured_state[1], s=0.1,color='red')


        # Plot real constraint
        x_c = np.linspace(-1,10,2000)
        y_c = (10-x_c)/2
        line1.set_data(x_c, y_c)

        # Plot tightened pseudo constraints
        path_g_z = os.path.join(path_dataset,"constraint_tightening","tightened_constraints","g_z_k_"+str(k)+".npy")
        curr_tight_first_constr = np.load(path_g_z)[0]
        print(f"curr_tight_first_constr:\n{curr_tight_first_constr}")
        curr_y_c = (curr_tight_first_constr-x_c)/2
        line2.set_data(x_c, curr_y_c)

        # Plot reference trajectory
        line3.set_data(ref_traj[0, 0:len_traj], ref_traj[1, 0:len_traj])

        ax_x_0.set_xlim(-1.3, 6.3)
        ax_x_0.set_ylim(-0.3, 4.5)

        # Plot distribution Delta x
        path_estim_pdf = os.path.join(path_dataset,"disturbance_estimation","disturbance_distribution_x_0","estim_pdf_k_"+str(k)+".npy")
        estimated_pdf = np.load(path_estim_pdf)
        resize_estimated_pdf = estimated_pdf*number_eval_points/(interv_max-interv_min)
        line_est_pdf_x.set_data(pdf_interval,resize_estimated_pdf)

        line_true_pdf_x.set_data(pdf_interval,true_pdf_x)
        ax_distr_x.set_xlim(-0.1,0.3)
        ax_distr_x.set_ylim(0,max(np.max(true_pdf_x),np.max(resize_estimated_pdf))+1)

        # Plot distribution Delta y
        path_estim_pdf = os.path.join(path_dataset,"disturbance_estimation","disturbance_distribution_x_1","estim_pdf_k_"+str(k)+".npy")
        estimated_pdf = np.load(path_estim_pdf)
        resize_estimated_pdf = estimated_pdf*number_eval_points/(interv_max-interv_min)
        line_est_pdf_y.set_data(pdf_interval,resize_estimated_pdf)

        line_true_pdf_y.set_data(pdf_interval,true_pdf_y)
        ax_distr_y.set_xlim(-0.1,0.4)
        ax_distr_y.set_ylim(0,max(np.max(true_pdf_y),np.max(resize_estimated_pdf))+1)



        # # Plot nominal prediction horizon
        # path_est_trajectory = os.path.join(path_dataset,"mpc","optimal_trajectories","trajectory_k_"+str(k)+".npy")
        # est_traj = np.load(path_est_trajectory)
        # nominal_pred_hor_x = est_traj[26::4]
        # nominal_pred_hor_y = est_traj[27::4]
        # line4.set_offsets(np.hstack([nominal_pred_hor_x,nominal_pred_hor_y]))

        # # Plot prediction horizon
        # # Adjust nominal prediction horizon with value of expected disturbance
        # exp_dist_x1 = np.dot(estimated_pdf,pdf_interval)
        # print(exp_dist_x1)
        # pred_hor_x1 = nominal_pred_hor_x1.copy()
        # for i in range(1,11):
        #     pred_hor_x1[(i-1):i] += i*exp_dist_x1
        # line5.set_offsets(np.hstack([idx_pred_horion,pred_hor_x1]))



        if k==0 or k==50 or k==100 or k==150 or k==200 or k==250 or k==300 or k==399 or k==400 or k==449 or k==499:
            path_cur_plot = os.path.join(path_plots,name_dataset+"_k_"+str(k)+".pdf")
            fig.savefig(path_cur_plot, format="pdf")#, bbox_inches="tight")


        # return line1, line2, line3, line4, ax_x_0, line_est_pdf_x, line_true_pdf_x, ax_distr_x, line_weights,bar_b_coeff,bar_b_coeff_text



    anim = animation.FuncAnimation(
        fig, animate, frames=range(0, len_traj), interval=100,repeat=False)

    # path_store_animation = os.path.join(path_dataset,"animation_"+name_dataset+".mp4")
    # writervideo = animation.FFMpegWriter(fps=10)
    # anim.save(path_store_animation, writer=writervideo)

    plt.show()

if __name__ == "__main__":
    animate_dataset()