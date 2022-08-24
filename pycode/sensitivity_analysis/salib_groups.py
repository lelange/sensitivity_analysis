from SALib.sample import saltelli
from SALib.analyze import sobol
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
path_input_factor_groups = 'data/input_factor_groups.pkl'

with open(path_simulation_result, 'rb') as f:
    size = pickle.load(f)
    input_factor_names = pickle.load(f)
    distributions = pickle.load(f)
    static_params = pickle.load(f)

with open(path_input_factor_groups, 'rb') as f:
    input_factor_group_dict = pickle.load(f)

problem = {
    'groups': [input_factor_group_dict[name] for name in input_factor_names],
    'num_vars': len(input_factor_names), 
    'names': input_factor_names,
    'bounds': [[distributions[i].getA(), distributions[i].getB()] for i in range(len(input_factor_names))]}

# generate samples
param_values = saltelli.sample(problem, 1024)

print(param_values.shape)


# start = time.time()
# Y = generate_output_daywise(param_values, input_factor_names, static_params)
# end = time.time()
# simulation_time = end - start
# print(f"Simulation run for {simulation_time} s.")

