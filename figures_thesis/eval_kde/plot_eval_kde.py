import os
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

mean=0
std_dev=1
num_runs=1000
num_max_samples=1000

path_store_animation = os.path.join(os.getcwd(),"figures_thesis","eval_kde","eval_kde_gaussian_mean_"+str(mean)+"_std_"+str(std_dev)+"_runs_"+str(num_runs)+".npy")
square_error_storage = np.load(path_store_animation)

x=list(range(2,num_max_samples+1))

plt.plot(x,np.sum(square_error_storage[:,1::],0)/num_runs)
plt.show()