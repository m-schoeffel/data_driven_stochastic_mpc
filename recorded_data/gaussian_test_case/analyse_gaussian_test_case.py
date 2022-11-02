import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

name_dataset = "gaussian_test_case"
name_ref_traj = "ref_traj_test_case.csv"
len_traj = 1900

current_wd = os.getcwd()
path_dataset = os.path.join(current_wd,"recorded_data",name_dataset)

start = 200
end = 1900
count_constraint_violation = 0

k_last_constraint_violation = -50

measured_state_storage = np.zeros(end-start)
for k in range(start,end):
    path_measured_state = os.path.join(path_dataset,"mpc","measured_states","real_state_k_"+str(k)+".npy")
    measured_state = np.load(path_measured_state)
    measured_state_storage[k-start] = measured_state[0]
    if measured_state[0]>2:
        if k-1 != k_last_constraint_violation:
            count_constraint_violation += 1
        k_last_constraint_violation = k


print(f"Number of total constraint violation: {count_constraint_violation}")
print(f"Percentage of constraint violation: {count_constraint_violation*100/(end-start)}")
print(f"Average value x1: {np.sum(measured_state_storage)/(end-start)}")


plt.plot(measured_state_storage)
plt.show()
