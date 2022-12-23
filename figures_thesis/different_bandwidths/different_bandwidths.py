import os
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import matplotlib.font_manager as font_manager


plt.rc('font', size=8)
# plt.rc('text', usetex=True)
plt.rc('font', family='serif')
without_serif= font_manager.FontProperties(family='sans-serif',
                               style='normal', size=7.5)
cm = 1/2.54  # centimeters in inches
my_figsize = (6.5*cm, 4*cm)
fig = plt.figure(figsize=my_figsize)

path_store_plot = os.path.join(os.getcwd(),"figures_thesis","different_bandwidths","plots_db")

cur_bandwith=0.1
datapoints = [-2,-1.5,-1.25,0.5,0,2]

x_eval = np.linspace(-4, 4, 4001)

distr_stor = list()

def gauss_kernel(bandwidth,datapoint):
    kernel_func = 1/(bandwidth*np.sqrt(2*np.pi))*np.exp(-0.5*np.power((x_eval-datapoint)/bandwidth,2))
    return kernel_func


for datapoint in datapoints:

    cur_distr = gauss_kernel(cur_bandwith,datapoint)
    distr_stor.append(cur_distr/len(datapoints))

# Plot kernel functions
for i,dist in enumerate(distr_stor):
    if i is 0:
        plt.plot(x_eval,dist,color="blue",label="Kernel functions")
    else:
        plt.plot(x_eval,dist,color="blue")

plt.plot(color="blue",label="Kernel functions")

# Plot estimated distribution
total_dist = distr_stor[0]
for i in range(1,len(distr_stor)):
    total_dist += distr_stor[i]

plt.plot(x_eval,total_dist,color="orange",label="Estimated distribution")

plt.scatter(datapoints,np.zeros(len(datapoints)),marker="x",color="black",label="Datapoints")

plt.legend(loc="upper right",prop=without_serif)

path_cur_plot = os.path.join(path_store_plot,"plot_diff_bandwidth_mean_bandwidth_"+str(cur_bandwith)+".pdf")
plt.savefig(path_cur_plot, format="pdf")

plt.show()