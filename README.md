# Data Driven Stochastic MPC with Online Disturbance Estimation and Constraint Tightening

The DDSMPC control scheme implemented in this repository corresponds to the Master Thesis "Data Driven Stochastic MPC with Online Disturbance Estimation and Constraint Tightening" of Matthias Schöffel and is currently developed.

## Overview

The implemented algorithm takes concepts from Stochastic Model Predictive Control (SMPC) and trajectory-based representations of LTI-Systems.

![Alt text](figures_thesis/overview/High_Level_DDSMPC.png?raw=true "Title")

### SMPC
In MPC an optimal control problem is iteratively solved for a finite horizon. Informally MPC finds the optimal next input by comparing the predicted trajectories of different control sequences and then applying the first input of the optimal input sequence. More information on MPC can be found here: [MPC](https://de.wikipedia.org/wiki/Model_Predictive_Control)

One of the big advantages of SMPC is the ability to enforce constraints on both the states and the input. In SMPC these constraints only have to be satisfied up to a specified probability. More information about SMPC can be found here: [SMPC](https://web.stanford.edu/class/ee364b/lectures/stoch_mpc_slides.pdf)

![Alt text](figures_thesis/overview/Low_Level_DDSMPC.png?raw=true "Title")

