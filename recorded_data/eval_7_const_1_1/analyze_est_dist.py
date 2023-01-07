from ctypes import alignment
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.font_manager as font_manager


name_dataset = "eval_7_const_1_1"
len_traj = 450

true_mean = 0
true_std_dev = 0.1

current_wd = os.getcwd()
path_dataset = os.path.join(current_wd,"recorded_data",name_dataset)

plt.rc('font', size=8)
# plt.rc('text', usetex=True)
plt.rc('font', family='serif')
without_serif= font_manager.FontProperties(family='sans-serif',
                               style='normal', size=7.5)
cm = 1/2.54  # centimeters in inches
my_figsize = (15*cm, 8*cm)
fig = plt.figure(figsize=my_figsize)
fig.tight_layout()

