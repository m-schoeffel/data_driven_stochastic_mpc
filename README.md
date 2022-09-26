# Data Driven Stochastic MPC with Online Disturbance Estimation and Constraint Tightening

The DDSMPC control scheme implemented in this repository corresponds to the Master Thesis "Data Driven Stochastic MPC with Online Disturbance Estimation and Constraint Tightening" of Matthias Schöffel and is currently under development.
The implemented algorithm adapts to the environment it is placed in and is able to satisfy constraints up to a specified probability level in the face of time-varying disturbances.

## Background

The implemented algorithm takes concepts from Stochastic Model Predictive Control (SMPC), trajectory-based representations of LTI-Systems and Kernel Density Estimation (KDE).

![Alt text](figures_thesis/overview/High_Level_DDSMPC.png?raw=true "Title")

### Stochastic Model Predictive Control
In MPC an optimal control problem is iteratively solved for a finite horizon. Informally MPC finds the optimal next input by comparing the predicted trajectories of different control sequences and then applying the first input of the optimal input sequence. More information on MPC can be found here: [MPC](https://de.wikipedia.org/wiki/Model_Predictive_Control)

One of the big advantages of SMPC is the ability to enforce constraints on both the states and the input. In SMPC these constraints only have to be satisfied up to a specified probability. More information about SMPC can be found here: [SMPC](https://web.stanford.edu/class/ee364b/lectures/stoch_mpc_slides.pdf)

### Trajectory-Based Representation of LTI-Systems
MPC controllers need a model of the system dynamics to predict finite horizon trajectories based on different control sequences. Traditionally the system dynamics of MPC controllers are specified by state-space equations. An alternative method is to only use a single input-state trajectory to specify the system dynamics. This method is used in the implemented Data Driven MPC algorithm of this repository.

### Kernel Density Estimation
Kernel Desity Estimation estimates probabilistic distributions based on samples of the distribution. A kernel function is placed on the position of every sample. The estimation of the distribution is created by summing up the kernel functions and normalizing the result. In DDSMPC scheme presented here a discount factor is introduced to discount older samples.

## Main Modules
The control scheme features three main modules: [Data-Driven MPC](data_driven_mpc/), [Disturbance Estimation](disturbance_estimation/) and [Constraint Tightening](constraint_tightening/).

### Data



![Alt text](figures_thesis/overview/Low_Level_DDSMPC.png?raw=true "Title")

