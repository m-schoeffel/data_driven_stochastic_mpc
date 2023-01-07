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
my_figsize = (10*cm, 6*cm)

fig = plt.figure(figsize=my_figsize)
fig.tight_layout()


data_points = np.array([8,-7,-2 ,3, 0])
weights = np.array([0.7,0.9,1.2,1.6,2.1])
weights = weights/np.sum(weights)
y_data_points = np.zeros(len(data_points))


x = np.linspace(-13, 17, 10000)

# Bandwith
h = 2


def gauss_kernel(x, x_i):

    prefactor = 1/(h*np.sqrt(2*np.pi))

    exponent = -0.5*np.power((x-x_i)/h, 2)

    result = prefactor * np.exp(exponent)

    return result


y_kernel_functions = np.zeros([len(x), len(data_points)])

for i, point in enumerate(data_points):
    y_kernel_functions[:, i] = weights[i] * gauss_kernel(x, point)


# Only normalize if no weights used
# y_kernel_functions = y_kernel_functions/len(data_points)

# # for i, point in enumerate(y_kernel_functions):

# plt.title(
#     r"\TeX\ is Number "r"$\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!")

plt.plot(x, np.sum(y_kernel_functions, 1), color="orange", lw=0.8,label="Estim. Distr. $f_{est}(\mathbf{d})$")

plt.plot(x, y_kernel_functions[:,0], label="Kernel functions",linestyle='--',color="blue", lw=0.8)
plt.plot(x, y_kernel_functions[:,1:],linestyle='--',color="blue", lw=0.8)

plt.scatter(data_points, y_data_points, marker='x',
            linewidths=2, color="green", label="Disturbances $\mathbf{d}_i$")

plt.xlabel("$\mathbf{d}$",usetex=True)
plt.ylabel("$f_{est}(\mathbf{d})$",usetex=True)

plt.legend(loc='upper right')

path_dataset = os.path.join(os.getcwd(),"figures_thesis","kde_examples","final_presentation_kde_example_h_"+str(h)+".png")
fig.savefig(path_dataset, format="png", bbox_inches="tight", dpi=1000)

# plt.show()
