import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime, date
import time
import openturns as ot

import pickle
import json
from utils_SA import simulate_model, generate_output_daywise
import openturns as ot
import os
import argparse

from varstool import VARS, Model

path_simulation_result = 'Studies/Sobol_MC_1_InfectedDead.pkl'
path_data = 'data/worldometer_data.txt'

with open(path_simulation_result, 'rb') as f:
    size = pickle.load(f)
    input_factor_names = pickle.load(f)
    distributions = pickle.load(f)
    static_params = pickle.load(f)

parameters = {}
for i in range(len(input_factor_names)):
    parameters[input_factor_names[i]] = [distributions[i].getA(), distributions[i].getB()]

experiment = VARS(
                parameters     = parameters,
                num_stars      = 100,
                delta_h        = 0.01,
                ivars_scales   = (0.1, 0.3, 0.5),
                sampler        = 'lhs',
                seed           = 123456789,
                bootstrap_flag = False,
                bootstrap_size = 100,
                bootstrap_ci   = 0.95,
                grouping_flag  = False,
                #num_grps = 3,
                report_verbose = True,
                )

save_star = False
run_model = True

if save_star:
    star_points = experiment.generate_star()
    star_points.to_csv('Studies/VARS/star_points.csv')
if run_model:

    with open("Studies/VARS/star_points.csv") as f:
        ncols = len(f.readline().split(','))
    print("Ncols: ", ncols, len(input_factor_names))
    input_data = np.loadtxt("Studies/VARS/star_points.csv", delimiter=',', skiprows=1, usecols=range(3,ncols))
    print(input_data.shape)
    static_params["output_index"] = [3]

    start = time.time()
    output_data = generate_output_daywise(input_data, input_factor_names, static_params)
    end = time.time()
    simulation_time = end - start
    print(f"Simulation run for {simulation_time} s.")

    print(output_data.shape)

    csv_input = pd.read_csv("Studies/VARS/star_points.csv")
    # save last day
    csv_input['output max infected'] = np.max(output_data, axis = 1)
    csv_input.to_csv('Studies/VARS/star_points_with_output_infected.csv', index = False )
    print("Saved to file.")


ex_modelframe = pd.read_csv('Studies/VARS/star_points_with_output_infected.csv', index_col=[0, 1, 2])
#print(type(ex_modelframe['output max dead'].values))
#print(ex_modelframe.index.get_level_values(-1))
#print(ex_modelframe.index.get_level_values(-2))
#print(type(ex_modelframe.index.get_level_values(-1)))

experiment.run_offline(ex_modelframe)

with open('Studies/VARS/VARS_experiment_infected.pkl', 'wb') as f: 
    pickle.dump(experiment, f)

#experiment.ivars[cols].to_csv('Studies/VARS/SA_results.csv', index = False )

# Plot IVARS from Experiment 1
ivars_scale = 0.5 # Choose the scale range of interest, e.g., 0.1, 0.3, or 0.5

cols = experiment.parameters.keys()                     

fig_bar = plt.figure(figsize=(15,10))
plt.gca().bar(cols, experiment.ivars.loc[pd.IndexSlice[ ivars_scale ]][cols], color='gold')
plt.gca().set_title (r'Integrated variogram Across a Range of Scales', fontsize = 10)
plt.gca().set_ylabel(r'IVARS-50 (Total-Variogram Effect)', fontsize = 13)
plt.gca().set_xlabel(r'Model Parameter', fontsize=6)
plt.gca().tick_params(labelrotation=90)
plt.gca().grid()
plt.gca().set_yscale('linear')
plt.tight_layout()
plt.savefig("latex_plots/VARS_50_infected.png")
plt.show()

fig_bar = plt.figure(figsize=(15,10))
plt.gca().bar(cols, experiment.ivars.loc[pd.IndexSlice[ ivars_scale ]][cols], color='gold')
plt.gca().set_title (r'Integrated variogram Across a Range of Scales $[log-scale]$', fontsize = 10)
plt.gca().set_ylabel(r'IVARS-50 (Total-Variogram Effect)', fontsize = 13)
plt.gca().set_xlabel(r'Model Parameter', fontsize=6)
plt.gca().tick_params(labelrotation=90)
plt.gca().grid()
plt.gca().set_yscale('log')
plt.tight_layout()

plt.savefig("latex_plots/VARS_50_log_infected.png")
plt.show()

# Directional Variograms

variograms1 = experiment.gamma.unstack(0)[cols].copy()
print(type(variograms1))
plotting_scale = 0.5 # any number between delta_h and one.

matrix_y = variograms1.loc[variograms1.index <= plotting_scale].to_numpy()
column_x = variograms1.loc[variograms1.index <= plotting_scale].index.to_numpy()
matrix_x = np.tile(column_x, (matrix_y.shape[1], 1)).T

fig_cdf = plt.figure(figsize=(10,5))
plt.gca().plot(matrix_x, matrix_y )
plt.gca().set_title (r'Directional Variogram', fontsize = 15)
plt.gca().set_ylabel(r'$γ(h)$', fontsize = 13)
plt.gca().set_xlabel(r'$h$ (perturbation scale)', fontsize=13)
plt.gca().set_yscale('linear')
plt.gca().legend (cols, loc='upper left', fontsize = 10)
plt.gca().grid()

fig_cdf = plt.figure(figsize=(10,5))
plt.gca().plot(matrix_x, matrix_y )
plt.gca().set_title (r'Directional Variogram $[log-scale]$', fontsize = 15)
plt.gca().set_ylabel(r'$γ(h)$', fontsize = 13)
plt.gca().set_xlabel(r'$h$ (perturbation scale)', fontsize=13)
plt.gca().set_yscale('log')
plt.gca().legend (cols, loc='lower right', fontsize = 10)
plt.gca().grid()

plt.savefig("latex_plots/VARS_directional_infected.png")
plt.show()

