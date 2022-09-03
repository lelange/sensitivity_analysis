import memilio.simulation.secir as secir
import numpy as np
import pandas as pd

from datetime import datetime, date
import time

from utils_SA import simulate_model, generate_output_daywise_one_factor

import openturns as ot
import os
import pickle
import argparse

'''

# define static parameters of the model

# Define Comartment names
compartments = ['Susceptible', 'Exposed', 'Carrier', 'Infected', 'Hospitalized', 'ICU', 'Recovered', 'Dead']
# Define age Groups
groups = ['0-4', '5-14', '15-34', '35-59', '60-79', '80+']
# Define population of age groups
populations = [40000, 70000, 190000, 290000, 180000, 60000] 

days = 100 # number of days to simulate
start_day = 1
start_month = 1
start_year = 2019
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
    'output_index' : [compartments.index("Dead")]
}

static_params["output_operation"]= "all" #"max"

# define input factors and their distributions
# TODO define "incubation_time" and "serial_interval" later
# todo max_rist_of_infection_from symptomatic -> twice rist_of_infection...

params_not_age_dependent = [
    # "incubation_time", 
    "serial_interval", 
    "infectious_mild_time", 
    "hospitalized_to_ICU_time", 
    # "relative_carrier_infectability",
    "risk_of_infection_from_symptomatic",
    "max_risk_of_infection_from_symptomatic",
]

params_not_age_dependent_short = [
    # r"t_{\textnormal{inc}}"
    r"t_{\textnormal{serint}}"
    r"$T_I^R$",
    r"$T_H^U$",
    # r"rel. C infec.",  
    r"$\tilde{\beta}_{\textnormal{min}}$",
    r"$\tilde{\beta}_{\textnormal{max}}$",
]

dist_not_age_dependent = [
    # ot.Uniform(5.2, 5.2)
    ot.Uniform(3.935, 4.6), #0.5 * 2.67 + 0.5 * 5.2; 0.5 * 4.00 + 0.5 * 5.2;
    ot.Uniform(5.6, 8.4), 
    ot.Uniform(3, 7), 
    # ot.Uniform(1, 1), 
    ot.Uniform(0.1, 0.3), 
    ot.Uniform(0.3, 0.5), 
]

params_damping = [
    'damping_coeff_home',
    'damping_coeff_school',
    'damping_coeff_work',
    'damping_coeff_other',

    'damping_time_home',
    'damping_time_school',
    'damping_time_work',
    'damping_time_other',
]

dist_damping = [
    # damping coefficient
    ot.Uniform(0.0, 1.0),
    ot.Uniform(0.0, 1.0),
    ot.Uniform(0.0, 1.0),
    ot.Uniform(0.0, 1.0),
    # day when damping starts
    ot.Uniform(0.0, days),
    ot.Uniform(0.0, days),
    ot.Uniform(0.0, days),
    ot.Uniform(0.0, days),
]

params_transition_duration = [   
    "hospitalized_to_home_time", 
    "home_to_hospitalized_time",
    "ICU_to_home_time", 
    "ICU_to_death_time"
]

params_transition_duration_ages = [s + f"_{i}" for s in params_transition_duration for i in range(len(groups))]

params_transition_duration_short = [      
    r"$T_H^R$", 
    r"$T_I^H$",    
    r"$T_U^R$", 
    r"$T_U^D$",
]

dist_transition_duration_ages = [ 
    # distributions of the parameters given in params_transition_duration list, one line contains the six age groups
    ot.Uniform(4, 6), ot.Uniform(4, 6), ot.Uniform(5, 7), ot.Uniform(7, 9), ot.Uniform(9, 11), ot.Uniform(13, 17),
    ot.Uniform(9, 12), ot.Uniform(9, 12), ot.Uniform(9, 12), ot.Uniform(5, 7), ot.Uniform(5, 7), ot.Uniform(5, 7),
    ot.Uniform(5, 9), ot.Uniform(5, 9), ot.Uniform(5, 9), ot.Uniform(14, 21), ot.Uniform(14, 21), ot.Uniform(10, 15),
    ot.Uniform(4, 8), ot.Uniform(4, 8), ot.Uniform(4, 8), ot.Uniform(15, 18), ot.Uniform(15, 18), ot.Uniform(10, 12)
]

initial_numbers_comp = [
    "init_exposed", 
    "init_carrier", 
    "init_infected", 
    "init_hospitalized", 
    "init_ICU", 
    "init_recovered",
    "init_dead"
]

dist_initial_numbers_comp = [
    ot.Uniform(50, 150), 
    ot.Uniform(25, 75), 
    ot.Uniform(10, 30),  
    ot.Uniform(10, 30),
    ot.Uniform(0, 20),
    ot.Uniform(0, 20),
    ot.Uniform(0, 1)
]

params_transition_probabilities = [ 
    "infection_probability_from_contact", 
    "asymptotic_cases_per_infectious", 
    "hospitalized_cases_per_infectious", 
    "ICU_cases_per_hospitalized",
    "deaths_per_ICU", 
]

params_transition_probabilities_ages = [s + f"_{i}" for s in params_transition_probabilities for i in range(len(groups))]


params_transition_duration_short = [  
    r"\rho", 
    r"$\mu_C^R$", 
    r"$\mu_I^H$", 
    r"$\mu_H^U$",
    r"$\mu_U^D$", 
]

dist_transition_probabilities = [
    ot.Uniform(0.02, 0.04), ot.Uniform(0.05, 0.07), ot.Uniform(0.05, 0.07), ot.Uniform(0.05, 0.07), ot.Uniform(0.08, 0.1), ot.Uniform(0.15, 0.2),
    ot.Uniform(0.2, 0.3), ot.Uniform(0.2, 0.3), ot.Uniform(0.15, 0.25), ot.Uniform(0.15, 0.25), ot.Uniform(0.15, 0.25), ot.Uniform(0.15, 0.25),
    ot.Uniform(0.006, 0.009), ot.Uniform(0.006, 0.009), ot.Uniform(0.015, 0.023), ot.Uniform(0.049, 0.074), ot.Uniform(0.15, 0.18), ot.Uniform(0.2, 0.25),
    ot.Uniform(0.05, 0.1), ot.Uniform(0.05, 0.1), ot.Uniform(0.05, 0.1), ot.Uniform(0.1, 0.2), ot.Uniform(0.25, 0.35), ot.Uniform(0.35, 0.45),
    ot.Uniform(0.0, 0.1), ot.Uniform(0.0, 0.1), ot.Uniform(0.1, 0.18), ot.Uniform(0.1, 0.18), ot.Uniform(0.3, 0.5), ot.Uniform(0.5, 0.7),
]

input_factor_names = params_not_age_dependent \
                    + params_damping \
                    + params_transition_duration_ages \
                    + params_transition_probabilities_ages \
                    + initial_numbers_comp
coll = dist_not_age_dependent \
    + dist_damping \
    + dist_transition_duration_ages \
    + dist_transition_probabilities \
    + dist_initial_numbers_comp

dimension = len(input_factor_names)

size = 1000 

'''
from inputFactorSpace import input_factor_names, coll

parser = argparse.ArgumentParser(description='Setup of the experiment parameters for sampling to visualize S_i.')
parser.add_argument('--MC_size_outer', '-no', help="outer sample size", type=int, default = 10)
parser.add_argument('--MC_size_inner', '-ni', help="inner sample size", type=int, default = 100)
parser.add_argument('--output_index', '-oi', help="index of compartment(s) for model output", type= int, nargs='+')

args = parser.parse_args()
N_inner = args.MC_size_inner
N_outer = args.MC_size_outer
output_index = args.output_index

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
static_params["output_operation"]= "all" #"max"

input_factor_i = 'deaths_per_ICU_5' #'serial_interval'
i = input_factor_names.index(input_factor_i)

##set coll ot distribution of X_{~i}
# remove name from list
input_factor_names.pop(i)
# distribution of X_{~i}, pop returns distribution of factor i
dist_i = coll.pop(i)

inputDistribution = ot.ComposedDistribution(coll)
inputDistribution.setDescription(input_factor_names)

ot.RandomGenerator.SetSeed(0)
# create random points for all factors except i
# random point for for all factors except i
N_outer = 10
MCexperiment = ot.MonteCarloExperiment(inputDistribution, N_outer)

MCinputDesign = MCexperiment.generate()
MCinputDesign.setDescription(input_factor_names)

N_inner = 100
#N = len(MCinputDesign)

print(f"Computed {N_outer} MC samples for all but input factor i.")

outs = []
for n in range(N_outer):
    fixed_params_n = dict(zip(input_factor_names, MCinputDesign[n]))
    MCexperiment_i = ot.MonteCarloExperiment(dist_i, N_inner)
    # sample input factor i
    MCinputDesign_i = MCexperiment_i.generate()
    print(len(MCinputDesign_i))
    print(type(MCinputDesign_i))
    # simulate model on different samples for i, other values fixed
    sim_out = generate_output_daywise_one_factor(MCinputDesign_i, input_factor_i, {**static_params, **fixed_params_n})
    # save value of last day
    outs.append(sim_out[:, -1])
print(f"Computed {N_outer} times {len(outs[0])} MC samples for input factor i.")

saving_path = f'Studies/MC_{input_factor_i}_{N_outer}_{N_inner}_{"".join(compartments[i] for i in output_index)}.pkl'
print(f"Study is saved to {saving_path}.")

with open(saving_path, 'wb') as f:
    pickle.dump(N_outer, f)
    pickle.dump(N_inner, f)
    pickle.dump(input_factor_names, f)
    pickle.dump(coll, f)
    pickle.dump(static_params, f)
    pickle.dump(input_factor_i, f)
    pickle.dump(dist_i, f)
    pickle.dump(outs, f)





