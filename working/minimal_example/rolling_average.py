import numpy as np
from matplotlib import pyplot as plt
import os,sys,glob
import shutil
import matplotlib as mpl

mpl.style.use('seaborn-deep')

def rolling_average(theta,amp):

    # specify a window width and step size for rolling average
    window_width = 2.0 # degrees
    step_size = 2.0 # degrees
    
    # find the start and end thetas
    t_start = np.min(theta)
    t_end = np.max(theta)

    window_centers = []
    amplitude_mean = []
    amplitude_std = []

    for t in np.arange(t_start,t_end+step_size,step_size):
        window_centers.append(t+window_width/2.0)

        # find he indices in the theta array where the angle falls in our window
        idx = np.where(np.logical_and(theta>=t,theta<t+window_width))[0]

        amplitude_mean.append(np.mean(amp[idx]))
        amplitude_std.append(np.std(amp[idx]))
                            
    # plot the raw data with low alpha
    plt.plot(theta,amp,'k.',alpha=0.1,markersize=1,label='raw data')

    # plot the average with standard deviation bars
    plt.errorbar(window_centers,amplitude_mean,amplitude_std,label='rolling average')

    plt.legend()
    plt.savefig('ra.png')
    plt.show()
        

# read one data set (subject 1, nondrusen) and separate theta and amplitude
raw_data = np.load('directionality_raw_data_nondrusen.npy')
theta = raw_data[:,0]
amp = raw_data[:,1]

rolling_average(theta,amp)


