import memilio.simulation.secir as secir
import numpy as np
import pandas as pd

from datetime import datetime, date
import time

from utils_SA import simulate_model, generate_output_daywise

import openturns as ot
import os
import pickle
import argparse

from inputFactorSpace import input_factor_names, distributions

parser = argparse.ArgumentParser(description='Setup of the experiment parameters.')
parser.add_argument('--MC_size', '-s', help="size of Monte Carlo experiment", type=int, default = 1000)
parser.add_argument('--output_index', '-oi', help="index of compartment(s) for model output", type= int, nargs='+')
parser.add_argument('--NPI_strength', '-npi', help="strength of NPI (weak, intermediate, strong)", type=str)

args = parser.parse_args()
MC_size = args.MC_size
output_index = args.output_index
NPI_scenario = args.NPI_strength

# Define Comartment names
compartments = ['Susceptible', 'Exposed', 'Carrier', 'Infected', 'Hospitalized', 'ICU', 'Recovered', 'Dead']
# Define age Groups
groups = ['0-4', '5-14', '15-34', '35-59', '60-79', '80+']
# Define population of age groups
populations = [40000, 70000, 190000, 290000, 180000, 60000] 

days = 100 # number of days to simulate
start_day = 18
start_month = 3
start_year = 2020
starting_day = (date(start_year, start_month, start_day) - date(start_year, 1, 1)).days
dt = 0.1
num_groups = len(groups)
num_compartments = len(compartments)

static_params = {
    'num_groups': num_groups, 
    'num_compartments': num_compartments,
    'populations': populations,
    'start_day' : (date(start_year, start_month, start_day) - date(start_year, 1, 1)).days,
    'days' : days,
    'dt' : dt,
    # which compartment's value should be outputed?
    'output_index' : output_index #compartments.index("Dead")
}
dimension = len(input_factor_names)

####
# set start day to same day
static_params['NPI_start_day_home'] = 10
static_params['NPI_start_day_school'] = 10
static_params['NPI_start_day_work'] = 10
static_params['NPI_start_day_other'] = 10
# remove from distribution
for factor in ['NPI_start_day_home', 'NPI_start_day_school', 'NPI_start_day_work', 'NPI_start_day_other', 'NPI_strength_school']:
    i = input_factor_names.index(factor)
    input_factor_names.pop(i)
    distributions.pop(i)

if NPI_scenario == 'weak':
### set up weak NPI
    NPI_strength_def = {
        'NPI_strength_home': ot.Uniform(0.0, 0.2),
        'NPI_strength_work': ot.Uniform(0.0, 0.2),
        'NPI_strength_other': ot.Uniform(0.0, 0.2),
    }
    static_params['NPI_strength_school'] = 0.25

if NPI_scenario == 'intermediate':
### set up intermediate NPI
    NPI_strength_def = {
        'NPI_strength_home': ot.Uniform(0.4, 0.6),
        'NPI_strength_work': ot.Uniform(0.4, 0.6),
        'NPI_strength_other': ot.Uniform(0.4, 0.6),
    }
    static_params['NPI_strength_school'] = 0.5

if NPI_scenario == 'strong':
### set up strong NPI
    NPI_strength_def = {
        'NPI_strength_home': ot.Uniform(0.8, 1.0),
        'NPI_strength_work': ot.Uniform(0.8, 1.0),
        'NPI_strength_other': ot.Uniform(0.8, 1.0),
    }
    static_params['NPI_strength_school'] = 1.0

# update distributions
for factor in ['NPI_strength_home', 'NPI_strength_work', 'NPI_strength_other']:
    i = input_factor_names.index(factor)
    distributions[i] = NPI_strength_def[factor]

inputDistribution = ot.ComposedDistribution(distributions)
inputDistribution.setDescription(input_factor_names)

print(inputDistribution.getDescription())
print(static_params)
print(input_factor_names)

size = MC_size 
start = time.time()
ot.RandomGenerator.SetSeed(2)
MC = ot.MonteCarloExperiment(inputDistribution, size)
print("Sample size: ", MC.getSize())
# generate samples from the input distribution
inputDesign = MC.generate()

input_names = inputDistribution.getDescription()
inputDesign.setDescription(input_factor_names)

sim_out = generate_output_daywise(inputDesign, input_factor_names, static_params)

end = time.time()
simulation_time = end - start
print(f"Simulation run for {simulation_time} s.")
#outputDesign = ot.Sample(sim_out)
saving_path = f'Studies/Sobol_NPIscenario_{NPI_scenario}_MC_{MC_size}_{"".join(compartments[i] for i in output_index)}.pkl'
print(f"Study is saved to {saving_path}.")

with open(saving_path, 'wb') as f:
    pickle.dump(size, f)
    pickle.dump(input_factor_names, f)
    pickle.dump(distributions, f)
    pickle.dump(static_params, f)
    pickle.dump(inputDesign, f)
    pickle.dump(sim_out, f)
    pickle.dump(simulation_time, f)
    pickle.dump(NPI_scenario, f)

