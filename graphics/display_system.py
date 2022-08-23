import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib

# Matplot config
matplotlib.rcParams['lines.linewidth'] = 3
matplotlib.rcParams['lines.markersize'] = 10
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.rc('font', size=20) #controls default text size

# Create subplots

figure, ax = plt.subplots(figsize=(14,15))

# Data Coordinates

x = np.linspace(0, 20, 80)
y = np.sin(x)

# GUI

plt.ion()

ax.scatter(1,1,linewidths=20)

plt.show()

for i in range(200):
    ax.plot([0,4],[1,5])
    ax.plot([0,4],[-1,3])
    ax.scatter(i/50,i/50)
    figure.canvas.draw()
    figure.canvas.flush_events()
    ax.clear()
    # time.sleep(0.01)


# #  Plot

# plot1, = ax.plot(x, y)

# # Labels

# plt.xlabel("X-Axis",fontsize=18)
# plt.ylabel("Y-Axis",fontsize=18)
# plt.show()

# for value in range(150):
#     update_y_value = np.sin(x-2.5*value)
    
#     plot1.set_xdata(x)
#     plot1.set_ydata(update_y_value)
    
#     figure.canvas.draw()
#     figure.canvas.flush_events()
#     time.sleep(0.01)


# Display

plt.show()