import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


cm = 1/2.54  # centimeters in inches

plt.rc('font', size=8)
# plt.rc('text', usetex=True)
plt.rc('font', family='serif')

without_serif= font_manager.FontProperties(family='sans-serif',
                                   style='normal', size=7.5)

cm = 1/2.54  # centimeters in inches
my_figsize = (7*cm, 4*cm)

fig = plt.figure(figsize=my_figsize)
fig.tight_layout()


data_points = np.array([-6, 2, 0, 3, 7])
y_data_points = np.zeros(len(data_points))


x = np.linspace(-13, 13, 10000)

# Bandwith
h = 4


def gauss_kernel(x, x_i):

    prefactor = 1/(h*np.sqrt(2*np.pi))

    exponent = -0.5*np.power((x-x_i)/h, 2)

    result = prefactor * np.exp(exponent)

    return result


y_kernel_functions = np.zeros([len(x), len(data_points)])

for i, point in enumerate(data_points):
    y_kernel_functions[:, i] = gauss_kernel(x, point)


y_kernel_functions = y_kernel_functions/len(data_points)

# # for i, point in enumerate(y_kernel_functions):

# plt.title(
#     r"\TeX\ is Number "r"$\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!")

plt.plot(x, y_kernel_functions, label="Kernel functions",linestyle='--',color="blue", lw=0.8)

plt.plot(x, np.sum(y_kernel_functions, 1), color="red", lw=0.8)

plt.scatter(data_points, y_data_points, marker='x',
            linewidths=2, color="green", label="Datapoints")

plt.xlabel("$s$",usetex=True)
plt.ylabel("$f(s)$",usetex=True)

# plt.legend(loc='upper left')

path_dataset = os.path.join(os.getcwd(),"figures_thesis","kde_examples","kde_example_h_"+str(h)+".pdf")
fig.savefig(path_dataset, format="pdf", bbox_inches="tight")

# plt.show()
