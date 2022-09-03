from SALib.sample import saltelli, finite_diff
from SALib.analyze import sobol, dgsm
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

path_simulation_result = 'Studies/Sobol_MC_1_Dead.pkl'
path_input_factor_groups = 'data/input_factor_groups.pkl'

use_groups = True
method = 'dgsm' #'variance' #
if method == 'dgsm':
    use_groups = False

# load information
with open(path_simulation_result, 'rb') as f:
    size = pickle.load(f)
    input_factor_names = pickle.load(f)
    distributions = pickle.load(f)
    static_params = pickle.load(f)

with open(path_input_factor_groups, 'rb') as f:
    input_factor_group_dict = pickle.load(f)

problem = {
    'num_vars': len(input_factor_names),    
    'names': input_factor_names,
    'bounds': [[distributions[i].getA(), distributions[i].getB()] for i in range(len(input_factor_names))]}

if use_groups:
    problem.update({'groups': [input_factor_group_dict[name] for name in input_factor_names]})

print(problem)
if method == 'variance':
    # generate samples
    N = 2**10 #up to 17
    param_values = saltelli.sample(problem, N)

    print(param_values.shape)

    start = time.time()
    Y = generate_output_daywise(param_values, input_factor_names, static_params)
    end = time.time()
    simulation_time = end - start
    print(f"Simulation run for {simulation_time} s.")

    with open(f'Studies/SAlib/SAlib_saltelli_groups_{N}.pkl', 'wb') as f: 
        pickle.dump(problem, f)
        pickle.dump(Y, f)
        pickle.dump(simulation_time, f)

if method == 'dgsm':
    # generate samples
    N = 1000
    param_values = finite_diff.sample(problem, N)

    print(param_values.shape)

    start = time.time()
    Y = generate_output_daywise(param_values, input_factor_names, static_params)
    end = time.time()
    simulation_time = end - start
    print(f"Simulation run for {simulation_time} s.")

    try:
        Si = dgsm.analyze(problem = problem, X = param_values, Y=Y, 
                        num_resamples=1000, conf_level=0.95, print_to_console=True, seed=None)
    except:
        "Error."

    saving_path = f'Studies/SAlib/SAlib_dgsm_{N}.pkl'
    
    with open(saving_path, 'wb') as f: 
        pickle.dump(problem, f)
        pickle.dump(param_values, f)
        pickle.dump(Y, f)
        pickle.dump(simulation_time, f)

    print(f"Study was saved to {saving_path}.")


