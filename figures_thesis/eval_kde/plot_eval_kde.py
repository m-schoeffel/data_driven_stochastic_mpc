import os
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import matplotlib.font_manager as font_manager


mean = 0
std_dev = 1
num_runs = 1000
num_max_samples = 1000

plt.rc('font', size=8)
# plt.rc('text', usetex=True)
plt.rc('font', family='serif')
without_serif = font_manager.FontProperties(family='sans-serif',
                                            style='normal', size=7.5)
cm = 1/2.54  # centimeters in inches
my_figsize = (15*cm, 8*cm)
fig = plt.figure(figsize=my_figsize)

path_store_animation = os.path.join(os.getcwd(), "figures_thesis", "eval_kde",
                                    "eval_kde_gaussian_mean_"+str(mean)+"_std_"+str(std_dev)+"_runs_"+str(num_runs)+".npy")
square_error_storage = np.load(path_store_animation)

# pdf was not normalized before recording squared error
# is fixed here
error_storage = np.sqrt(square_error_storage)
norm_error_storage = error_storage * (4-(-4))/4001
square_error_storage = np.power(norm_error_storage, 2)

x = list(range(6, num_max_samples+1))

plt.yscale("log")

mean_se = np.sum(square_error_storage[:, 5::], 0)/num_runs
plt.plot(x, mean_se, label="Mean square error $\mu$")

std_dev_se = np.std(square_error_storage[:, 5::], axis=0)
plt.fill_between(x, mean_se-std_dev_se, mean_se +
                 std_dev_se, color="orange", alpha=0.2)

plt.plot(x, mean_se-std_dev_se, color="orange",
         label="First standard deviation $\sigma$")
plt.plot(x, mean_se+std_dev_se, color="orange")

plt.xlabel("Number of samples $o$ used for KDE", usetex=True)
plt.ylabel("Square error $\epsilon_{\mathrm{square}}$", usetex=True)

print(std_dev_se[0:10])

plt.legend(loc="upper right", prop=without_serif)

plt.tight_layout()

path_cur_plot = os.path.join(
    os.getcwd(), "figures_thesis", "eval_kde", "eval_kde_runs_"+str(num_runs)+".pdf")
plt.savefig(path_cur_plot, format="pdf")

plt.show()
