import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from config import load_parameters
    
    # ------------------ Animate state sequence ------------------
def animate_state_sequence(state_storage):
    fig = plt.figure()
    ax = plt.axes(xlim=(-4, 4), ylim=(-4, 4))

    # prepare plots for joint constraints for x1 and x2
    # Todo: Make plottet constraints truly flexible
    constraints = load_parameters.load_constraints()
    G_x = np.array(constraints["G_x"])
    g_x = np.array(constraints["g_x"])
    x1_constr=np.ones(100)*g_x[0]
    y1_constr=np.linspace(-4,4,100)

    x2_constr=np.linspace(-4,4,100)
    y2_constr=np.ones(100)*g_x[1]


    def animate(i):
        ax.scatter(state_storage[0,i],state_storage[1,i])
        ax.scatter(0,0)

        ax.plot(x1_constr,y1_constr,color='blue',lw=10)
        ax.plot(x2_constr,y2_constr,color='blue',lw=10)

    anim = animation.FuncAnimation(fig, animate, interval=1000)

    plt.show()