import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# Matplot config
matplotlib.rcParams['lines.linewidth'] = 3
matplotlib.rcParams['lines.markersize'] = 10
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.rc('font', size=12)  # controls default text size
# plt.rc('axes', titlesize=10) #fontsize of the title
# plt.rc('axes', labelsize=10) #fontsize of the x and y labels
# plt.rc('xtick', labelsize=10) #fontsize of the x tick labels
# plt.rc('ytick', labelsize=10) #fontsize of the y tick labels
plt.rc('legend', fontsize=10)  # fontsize of the legend

cm = 1/2.54  # centimeters in inches
my_figsize = (16*cm, 8*cm)
fig = plt.figure(figsize=my_figsize)


data_points = np.array([-6, 2, 0, 3, 7])
y_data_points = np.zeros(len(data_points))


x = np.linspace(-10, 10, 10000)

# Bandwith
h = 2


def gauss_kernel(x, x_i):

    prefactor = 1/(h*np.sqrt(2*np.pi))

    exponent = -0.5*np.power((x-x_i)/h, 2)

    result = prefactor * np.exp(exponent)

    return result


y_kernel_functions = np.zeros([len(x), len(data_points)])

for i, point in enumerate(data_points):
    y_kernel_functions[:, i] = gauss_kernel(x, point)


y_kernel_functions = y_kernel_functions/len(data_points)

# for i, point in enumerate(y_kernel_functions):

plt.title(
    r"\TeX\ is Number "r"$\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!")

plt.plot(x, y_kernel_functions, label="Kernel functions")

plt.plot(x, np.sum(y_kernel_functions, 1), linestyle='--', label="hallo")

plt.scatter(data_points, y_data_points, marker='x',
            linewidths=15, label="Datapoints")

plt.xlabel("This is my x.")

plt.legend(loc='upper left')

plt.savefig("my_first_saved_plot.pdf", format="pdf", bbox_inches="tight")

plt.show()
