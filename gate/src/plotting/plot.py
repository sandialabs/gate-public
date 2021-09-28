import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
import csv
from mpl_toolkits.mplot3d import Axes3D


def load_data(csv_path):
    with open(csv_path, 'r') as f:
        data = csv.reader(f).__next__()

    n_states = int(data[0])
    n_steps = int(data[1])
    n_trajs = int(data[2])
    n_data_points = n_states * n_steps * n_trajs

    data = np.array(data[3 : n_data_points + 3], dtype=np.float64)
    data = data.reshape(n_trajs, n_steps, n_states)
    return n_states, n_steps, n_trajs, data


def plot_trajectory_subplots(n_states, n_steps, n_trajs, data, dt):
    t = [x * dt for x in range(n_steps)]

    n_rows = int(np.ceil(n_states / 3))
    n_cols = 3

    _, axs = plt.subplots(n_rows, 3)

    axs[n_rows - 1, 0].set_xlabel('t (s)')
    axs[n_rows - 1, 1].set_xlabel('t (s)')
    axs[n_rows - 1, 2].set_xlabel('t (s)')

    axs = axs.flat

    for state in range(n_states):
        for traj in range(n_trajs):
            x = data[traj, :, state]
            axs[state].plot(t, x)

        x_f_mean = np.mean(data[:, -1, state])
        x_f_std = np.std(data[:, -1, state])
        axs[state].set_ylabel(f'State {state}')
        axs[state].set_title(f'x_f_mean : {x_f_mean:.4f} \n x_f_std : {x_f_std:.4f}')
        axs[state].grid()   
    

    left  = 0.125  # the left side of the subplots of the figure
    right = 0.9    # the right side of the subplots of the figure
    bottom = 0.1   # the bottom of the subplots of the figure
    top = 0.9      # the top of the subplots of the figure
    wspace = 0.3   # the amount of width reserved for blank space between subplots
    hspace = 0.55   # the amount of height reserved for white space between subplots
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()


def plot_2d_trajectory_figures(n_states, n_steps, n_trajs, data, dt, plot_tuples_2d, x_y_title_labels_2d, trajectory_idx_cutoff=None):
    t = [x * dt for x in range(n_steps)]
    idx_count = 0
    for (x_idx, y_idx) in plot_tuples_2d:
        x = np.array([t for _ in range(n_trajs)]) if x_idx == 't' else data[:, :, x_idx]        
        y = data[:, :, y_idx]

        if trajectory_idx_cutoff is not None:
            x = np.array([x_new[:trajectory_idx_cutoff] for x_new in x])
            y = np.array([y_new[:trajectory_idx_cutoff] for y_new in y])

        plt.figure(figsize=(10,10))
        
        # the first shape of t and the second shape of y need to be the same
        # the downsample of t and the downsample of the second dimension of y need to match
        # in other words, the first shape of t and the first shape of y.T need to be the same
        # y is num rollouts by num timesteps, usually 500 x 5000 
        plt.plot(x.T, y.T)
        
        x_f_mean = np.mean(data[:, -1, y_idx])
        x_f_std = np.std(data[:, -1, y_idx])
        
        x_label, y_label, title_label = x_y_title_labels_2d[idx_count] 
        
        plt.xlabel(x_label)
        plt.title(title_label)
        plt.ylabel(y_label)
        plt.grid()
        
        idx_count = idx_count+1


def plot_3d_trajectory_figures(n_states, n_steps, n_trajs, data, dt, plot_tuples_3d):
    t = [x * dt for x in range(n_steps)]
    
    for (x_idx, y_idx, z_idx) in plot_tuples_3d:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        for traj in range(n_trajs):
            x = t if x_idx == 't' else data[traj, :, x_idx]
            y = t if y_idx == 't' else data[traj, :, y_idx]
            z = t if z_idx == 't' else data[traj, :, z_idx]
            ax.plot(x, y, z)
            ax.set_xlabel(f'State {x_idx}')
            ax.set_ylabel(f'State {y_idx}')
            ax.set_zlabel(f'State {z_idx}')


def plot_state_histograms(n_states, n_steps, n_trajs, data, trajectory_idx=None):
    # distribution of final state
    if trajectory_idx is None:
        idx = -1
    else:
        idx = trajectory_idx
    for x_idx in range(n_states):
        vals = data[:, idx, x_idx]
        fig = plt.figure(figsize=(10,10))
        
#        ax1 = fig.add_subplot(111)
        plt.grid(zorder=0)
        
        # Histogram:
        #https://stackoverflow.com/questions/38650550/cant-get-y-axis-on-matplotlib-histogram-to-display-probabilities
        # Bin it
        n, bin_edges = np.histogram(vals, 50)
        # Normalize it, so that every bins value gives the probability of that bin
        bin_probability = n/float(n.sum())
        # Get the mid points of every bin
        bin_middles = (bin_edges[1:]+bin_edges[:-1])/2.
        # Compute the bin-width
        bin_width = bin_edges[1]-bin_edges[0]
        # Plot the histogram as a bar plot
        plt.bar(bin_middles, bin_probability, width=bin_width, edgecolor='black', zorder=3)
        
        plt.xlabel(f'State {x_idx}')

    



def main(csv_path):
    subplots = False
    figures2d = True
    figures3d = False
    hist = False
    trajectory_idx=None #500

    n_states, n_steps, n_trajs, data = load_data(csv_path)

    print('n_states: ', n_states)
    print('n_steps: ', n_steps)
    print('n_trajs: ', n_trajs)
    print('data shape: ', data.shape)
    
    
    # mass-spring damper
#    dt = 0.01
#    plot_tuples_2d = [('t', 0)] #, (1,2)]
#    x_y_title_labels_2d = [('time (s)', 'x (m)', 'Position vs. Time')]
#    trajectory_idx_cutoff=None
    # dubins
    dt = 0.01
    plot_tuples_2d = [(0, 1)] #, (1,2)]
    x_y_title_labels_2d = [('x (m)', 'y (m)', 'y vs x')]
    plot_tuples_3d = None #[(0, 1, 2), ('t', 0, 1)]
    trajectory_idx_cutoff=200
    
    if subplots:
        plot_trajectory_subplots(n_states, n_steps, n_trajs, data, dt)
    if figures2d:
        plot_2d_trajectory_figures(n_states, n_steps, n_trajs, data, dt, plot_tuples_2d, x_y_title_labels_2d,trajectory_idx_cutoff)
    if figures3d:
        plot_3d_trajectory_figures(n_states, n_steps, n_trajs, data, dt, plot_tuples_3d)
    if hist:
        plot_state_histograms(n_states, n_steps, n_trajs, data, trajectory_idx)

    plt.show()


if __name__ == '__main__':
    # csv_path='../../trajectories/dubins/control_and_ics.csv'
    csv_path='../../trajectories/spring_mass_damper/spring_mass_damper_2021_04_20__16_53_30.csv'
    main(csv_path)
