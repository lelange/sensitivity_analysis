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

path_simulation_result = 'Studies/Sobol_MC_100000_InfectedDead.pkl'
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
                num_stars      = 50,
                delta_h        = 0.1,
                ivars_scales   = (0.1, 0.3, 0.5),
                sampler        = 'lhs',
                seed           = 123456789,
                bootstrap_flag = True,
                bootstrap_size = 100,
                bootstrap_ci   = 0.9,
                grouping_flag  = True,
                num_grps = 3,
                report_verbose = True,
                )



save_star = False
run_model = False

if save_star:
    star_points = experiment.generate_star()
    star_points.to_csv('Studies/VARS/star_points.csv')
if run_model:

    with open("Studies/VARS/star_points.csv") as f:
        ncols = len(f.readline().split(','))
    print("Ncols: ", ncols, len(input_factor_names))
    input_data = np.loadtxt("Studies/VARS/star_points.csv", delimiter=',', skiprows=1, usecols=range(3,ncols))
    print(input_data.shape)
    static_params["output_index"] = [7]

    start = time.time()
    output_data = generate_output_daywise(input_data, input_factor_names, static_params)
    end = time.time()
    simulation_time = end - start
    print(f"Simulation run for {simulation_time} s.")

    print(output_data.shape)

    csv_input = pd.read_csv("Studies/VARS/star_points.csv")
    # save last day
    csv_input['output max dead'] = output_data[:, -1]
    csv_input.to_csv('Studies/VARS/star_points_with_output.csv', index = False )
    print("Saved to file.")


ex_modelframe = pd.read_csv('Studies/VARS/star_points_with_output.csv', index_col=[0, 1, 2])
#print(type(ex_modelframe['output max dead'].values))
#print(ex_modelframe.index.get_level_values(-1))
#print(ex_modelframe.index.get_level_values(-2))
#print(type(ex_modelframe.index.get_level_values(-1)))

experiment.run_offline(ex_modelframe)

(experiment.output).to_csv('Studies/VARS/VARS_output.csv', index = False )

cols = experiment.parameters.keys()
experiment.ivars[cols].to_csv('Studies/VARS/SA_results.csv', index = False )

# Plot IVARS from Experiment 1
ivars_scale = 0.5 # Choose the scale range of interest, e.g., 0.1, 0.3, or 0.5

cols = experiment.parameters.keys()                     
fig_bar = plt.figure(figsize=(10,5))
plt.gca().bar(cols, experiment.ivars.loc[pd.IndexSlice[ ivars_scale ]][cols], color='gold')
plt.gca().set_title (r'Integrated variogram Across a Range of Scales', fontsize = 15)
plt.gca().set_ylabel(r'IVARS-50 (Total-Variogram Effect)', fontsize = 13)
plt.gca().set_xlabel(r'Model Parameter', fontsize=13)
plt.gca().tick_params(labelrotation=45)
plt.gca().grid()
plt.gca().set_yscale('linear')

fig_bar = plt.figure(figsize=(10,5))
plt.gca().bar(cols, experiment.ivars.loc[pd.IndexSlice[ ivars_scale ]][cols], color='gold')
plt.gca().set_title (r'Integrated variogram Across a Range of Scales $[log-scale]$', fontsize = 15)
plt.gca().set_ylabel(r'IVARS-50 (Total-Variogram Effect)', fontsize = 13)
plt.gca().set_xlabel(r'Model Parameter', fontsize=13)
plt.gca().tick_params(labelrotation=45)
plt.gca().grid()
plt.gca().set_yscale('log')
plt.show()

plt.savefig("plots/VARS.png")


