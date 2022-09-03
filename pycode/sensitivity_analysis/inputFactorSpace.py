import openturns as ot
from datetime import datetime, date
import numpy as np
import pandas as pd
import time
import pickle

'''
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

params_not_age_dependent = [
    "dummy",
    "incubation_time", 
    "serial_interval", 
    "infectious_mild_time", 
    "hospitalized_to_ICU_time", 
    "risk_of_infection_from_symptomatic",
    "max_risk_of_infection_from_symptomatic",
    "seasonality",
    "test_and_trace_capacity" # multiplier of #tnt_capacity = np.sum(populations)*100.0/100000*(10.0/7)
]

params_not_age_dependent_short = [
    "dummy",
    r"t_{\textnormal{inc}}"
    r"t_{\textnormal{serint}}"
    r"$T_I^R$",
    r"$T_H^U$", 
    r"$\tilde{\beta}_{\textnormal{min}}$",
    r"$\tilde{\beta}_{\textnormal{max}}$",
    r"k",
    "tt_capacity"
]

dist_not_age_dependent = [
    ot.Uniform(0, 1),
    ot.Uniform(4.1, 7.0), # before 5.2
    ot.Uniform(0.63, 0.87), #0.5 * 2.67 + 0.5 * 5.2; 0.5 * 4.00 + 0.5 * 5.2; before (3.935, 4.6)
    ot.Uniform(5.6, 8.4), 
    ot.Uniform(3, 7),  
    ot.Uniform(0.1, 0.3), 
    ot.Uniform(0.3, 0.5), 
    ot.Uniform(-0.3, 0.3), 
    ot.Uniform(0.8, 1.2)
]

params_damping = [
    'NPI_strength_home',
    'NPI_strength_school',
    'NPI_strength_work',
    'NPI_strength_other',
    #
    'NPI_start_day_home',
    'NPI_start_day_school',
    'NPI_start_day_work',
    'NPI_start_day_other',
]

dist_damping = [
    ot.Uniform(0.0, 1.0),
    ot.Uniform(0.0, 1.0),
    ot.Uniform(0.0, 1.0),
    ot.Uniform(0.0, 1.0),
    #
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

# only needed for parameter estimation and if the dataset starts after the beginning of the dynamics
initial_numbers_comp = [
    "init_exposed", 
    "init_carrier", 
    "init_infected", 
    #"init_hospitalized", 
    #"init_ICU", 
    #"init_recovered",
    #"init_dead"
]

dist_initial_numbers_comp = [
    ot.Uniform(0, 200), 
    ot.Uniform(0, 200), 
    ot.Uniform(120, 130),  
    #ot.Uniform(0, 1),
    #ot.Uniform(0, 1),
    #ot.Uniform(0, 0.1),
    #ot.Uniform(0, 0.001)
]

params_transition_probabilities = [ 
    "infection_probability_from_contact", 
    "relative_carrier_infectability",
    "asymptotic_cases_per_infectious", 
    "hospitalized_cases_per_infectious", 
    "ICU_cases_per_hospitalized",
    "deaths_per_ICU", 
]

params_transition_probabilities_ages = [s + f"_{i}" for s in params_transition_probabilities for i in range(len(groups))]

params_transition_duration_short = [  
    r"\rho", 
    r"rel. C infec.",
    r"$\mu_C^R$", 
    r"$\mu_I^H$", 
    r"$\mu_H^U$",
    r"$\mu_U^D$", 
]

dist_transition_probabilities = [
    ot.Uniform(0.02, 0.04), ot.Uniform(0.05, 0.07), ot.Uniform(0.05, 0.07), ot.Uniform(0.05, 0.07), ot.Uniform(0.08, 0.1), ot.Uniform(0.15, 0.2),
    ot.Uniform(0.5, 1.5), ot.Uniform(0.5, 1.5), ot.Uniform(0.5, 1.5), ot.Uniform(0.5, 1.5), ot.Uniform(0.5, 1.5), ot.Uniform(0.5, 1.5),
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

inputDistribution = ot.ComposedDistribution(coll)
inputDistribution.setDescription(input_factor_names)

lower_bounds = [coll[i].getA() for i in range(len(input_factor_names))]
upper_bounds = [coll[i].getB() for i in range(len(input_factor_names))]

'''
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


# define input factors and their distributions
# todo max_rist_of_infection_from symptomatic -> twice rist_of_infection...

params_not_age_dependent = [
    "dummy",
    "incubation_time", 
    "serial_interval", 
    "infectious_mild_time", 
    "hospitalized_to_ICU_time", 
    "risk_of_infection_from_symptomatic",
    "max_risk_of_infection_from_symptomatic",
    "seasonality",
    "test_and_trace_capacity" # multiplier of #tnt_capacity = np.sum(populations)*100.0/100000*(10.0/7)
]


params_not_age_dependent_category = [
    "helper",
    "virus related", 
    "virus related", 
    "transition time", 
    "transition time", 
    "virus related",
    "virus related",
    "virus related",
    "NPI related"
]

params_not_age_dependent_short = [
    "d",
    r"$t_{\textnormal{inc}}$",
    r"$t_{\textnormal{serint}}$",
    r"$T_I^R$",
    r"$T_H^U$", 
    r"$\tilde{\beta}_{I, \textnormal{min}}$",
    r"$\tilde{\beta}_{I, \textnormal{max}}$",
    r"k",
    "cap",
]

dist_not_age_dependent = [
    ot.Uniform(0, 1),
    ot.Uniform(4.1, 7.0), # before 5.2
    ot.Uniform(0.63, 0.87), #0.5 * 2.67 + 0.5 * 5.2; 0.5 * 4.00 + 0.5 * 5.2; before (3.935, 4.6)
    ot.Uniform(5.6, 8.4), 
    ot.Uniform(3, 7),  
    ot.Uniform(0.1, 0.3), 
    ot.Uniform(0.3, 0.5), 
    ot.Uniform(-0.3, 0.3), 
    ot.Uniform(0.8, 1.2)
]

params_damping = [
    'NPI_strength_home',
    'NPI_strength_school',
    'NPI_strength_work',
    'NPI_strength_other',
    #
    'NPI_start_day_home',
    'NPI_start_day_school',
    'NPI_start_day_work',
    'NPI_start_day_other',
]

params_damping_category = ["NPI related"]*len(params_damping)

params_damping_short = [
    r'$r_{\text{home}}$',
    r'$r_{\text{school}}$',
    r'$r_{\text{work}}$',
    r'$r_{\text{other}}$',
    #
    r'$t_{\text{home}}^r$',
    r'$t_{\text{school}}^r$',
    r'$t_{\text{work}}^r$',
    r'$t_{\text{other}}^r$',
]

dist_damping = [
    ot.Uniform(0.0, 1.0),
    ot.Uniform(0.0, 1.0),
    ot.Uniform(0.0, 1.0),
    ot.Uniform(0.0, 1.0),
    #
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
params_transition_duration_short = [      
    r"$T_H^R$", 
    r"$T_I^H$",    
    r"$T_U^R$", 
    r"$T_U^D$",
]

params_transition_duration_ages = [s + f"_{i}" for s in params_transition_duration for i in range(len(groups))]

params_transition_duration_ages_short = [s  for s in params_transition_duration_short for i in range(len(groups))]

params_transition_duration_ages_category = ["transition time"]*len(params_transition_duration_ages)



dist_transition_duration_ages = [ 
    # distributions of the parameters given in params_transition_duration list, one line contains the six age groups
    ot.Uniform(4, 6), ot.Uniform(4, 6), ot.Uniform(5, 7), ot.Uniform(7, 9), ot.Uniform(9, 11), ot.Uniform(13, 17),
    ot.Uniform(9, 12), ot.Uniform(9, 12), ot.Uniform(9, 12), ot.Uniform(5, 7), ot.Uniform(5, 7), ot.Uniform(5, 7),
    ot.Uniform(5, 9), ot.Uniform(5, 9), ot.Uniform(5, 9), ot.Uniform(14, 21), ot.Uniform(14, 21), ot.Uniform(10, 15),
    ot.Uniform(4, 8), ot.Uniform(4, 8), ot.Uniform(4, 8), ot.Uniform(15, 18), ot.Uniform(15, 18), ot.Uniform(10, 12)
]

# only needed for parameter estimation and if the dataset starts after the beginning of the dynamics
initial_numbers_comp = [
    "init_exposed", 
    "init_carrier", 
    "init_infected", 
    #"init_hospitalized", 
    #"init_ICU", 
    #"init_recovered",
    #"init_dead"
]

initial_numbers_comp_category = ["initial value"]*len(initial_numbers_comp)

initial_numbers_comp_short = [
    r"$\text{init}_E$", 
    r"$\text{init}_C$", 
    r"$\text{init}_I$", 
    #"init_hospitalized", 
    #"init_ICU", 
    #"init_recovered",
    #"init_dead"
]

dist_initial_numbers_comp = [
    ot.Uniform(0, 200), 
    ot.Uniform(0, 200), 
    ot.Uniform(120, 130),  
    #ot.Uniform(0, 1),
    #ot.Uniform(0, 1),
    #ot.Uniform(0, 0.1),
    #ot.Uniform(0, 0.001)
]

params_transition_probabilities = [ 
    "infection_probability_from_contact", 
    "relative_carrier_infectability",
    "asymptotic_cases_per_infectious", 
    "hospitalized_cases_per_infectious", 
    "ICU_cases_per_hospitalized",
    "deaths_per_ICU", 
]

params_transition_probabilities_ages = [s + f"_{i}" for s in params_transition_probabilities for i in range(len(groups))]

params_transition_probabilities_ages_category = ["transition probability"]*len(params_transition_probabilities_ages)
params_transition_probabilities_short = [  
    r"$\rho$", 
    r"$\tilde{\beta}_C$",
    r"$\mu_C^R$", 
    r"$\mu_I^H$", 
    r"$\mu_H^U$",
    r"$\mu_U^D$", 
]
params_transition_probabilities_ages_short = [s for s in params_transition_probabilities_short for i in range(len(groups))]

#print(params_transition_probabilities_ages_short)

dist_transition_probabilities_ages = [
    ot.Uniform(0.02, 0.04), ot.Uniform(0.05, 0.07), ot.Uniform(0.05, 0.07), ot.Uniform(0.05, 0.07), ot.Uniform(0.08, 0.1), ot.Uniform(0.15, 0.2),
    ot.Uniform(0.5, 1.5), ot.Uniform(0.5, 1.5), ot.Uniform(0.5, 1.5), ot.Uniform(0.5, 1.5), ot.Uniform(0.5, 1.5), ot.Uniform(0.5, 1.5),
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

input_factor_names_short = params_not_age_dependent_short \
                    + params_damping_short \
                    + params_transition_duration_ages_short \
                    + params_transition_probabilities_ages_short \
                    + initial_numbers_comp_short

age_dep = [False for i in params_not_age_dependent] \
                    + [False for i in params_damping] \
                    + [True for i in params_transition_duration_ages] \
                    + [True for i in params_transition_probabilities_ages] \
                    + [False for i in initial_numbers_comp]

categories = params_not_age_dependent_category \
            + params_damping_category \
            + params_transition_duration_ages_category \
            + params_transition_probabilities_ages_category \
            + initial_numbers_comp_category

print(len(input_factor_names))

distributions = dist_not_age_dependent \
    + dist_damping \
    + dist_transition_duration_ages \
    + dist_transition_probabilities_ages \
    + dist_initial_numbers_comp

name_dict = dict(zip(input_factor_names,input_factor_names_short))

group_list = np.zeros(len(input_factor_names))
group_names = [
    # compartment_dependent
    "init", 
    # location dependent
    "NPI_strength",
    "NPI_start",
    # age dependent
    "hospitalized_to_home_time",
    "home_to_hospitalized_time",
    "ICU_to_home_time",
    "ICU_to_death_time",
    "infection_probability_from_contact",
    "relative_carrier_infectability",
    "asymptotic_cases_per_infectious",
    "hospitalized_cases_per_infectious",
    "ICU_cases_per_hospitalized",
    "deaths_per_ICU",
    # other parameters
    'dummy',
    'incubation_time',
    'serial_interval',
    'infectious_mild_time',
    'hospitalized_to_ICU_time',
    'risk_of_infection_from_symptomatic',
    'seasonality',
    'test_and_trace_capacity'
]   

compartment = ["initial_infections"]
location = [
    "NPI_strength",
    "NPI_start",]
age = [
      "hospitalized_to_home_time",
    "home_to_hospitalized_time",
    "ICU_to_home_time",
    "ICU_to_death_time",
    "infection_probability_from_contact",
    "relative_carrier_infectability",
    "asymptotic_cases_per_infectious",
    "hospitalized_cases_per_infectious",
    "ICU_cases_per_hospitalized",
    "deaths_per_ICU",  
    ]
other = [
    'dummy',
    'incubation_time',
    'serial_interval',
    'infectious_mild_time',
    'hospitalized_to_ICU_time',
    'risk_of_infection_from_symptomatic',
    'seasonality', 
    'test_and_trace_capacity'
    ]      

group_ids = dict(zip(group_names, range(len(group_names))))
#print(group_ids)
groups = [name for factor_name in input_factor_names for name in group_names if (name in factor_name) ]
#print(groups)
#print(len(groups))

# factor is in which group
group_dict = {}
for i in range(len(input_factor_names)):
    group_dict[input_factor_names[i]] = groups[i]
    
problem = {  
    'names': input_factor_names,
    'bounds': [[distributions[i].getA(), distributions[i].getB()] for i in range(len(input_factor_names))]}

df_input_factors = pd.DataFrame({"Name": input_factor_names, 
                   "Short": input_factor_names_short, 
                   "ot dist": distributions, 
                   'bounds': [[distributions[i].getA(), distributions[i].getB()] for i in range(len(input_factor_names))],
                   'lower_bounds': [distributions[i].getA() for i in range(len(input_factor_names))],
                   'upper_bounds': [distributions[i].getB() for i in range(len(input_factor_names))],
                   "Age-dep": age_dep, 
                   "Group": [group_dict[name] for name in input_factor_names],
                   "Category": categories
                  })


df_fixing = pd.read_csv('results/dgsm_vi_important.csv', index_col = 0)
df_input_factors['fix_by_vi']=[df_fixing.loc[name, 'vi'] for name in input_factor_names]
df_input_factors = df_input_factors.set_index('Name', drop = True)

def fix_factors(df_input_factors, column_to_fix = 'fix_by_vi', fixed_value = 'mean'):
    dict_fixed_factors = {}
    if fixed_value == 'mean':
        dict_fixed_factors = df_input_factors[df_input_factors[column_to_fix]].loc[:, 'ot dist'].apply(lambda x : x.getMean()[0]).to_dict()
    if fixed_value == 'random':
        df_input_factors[df_input_factors[column_to_fix]].loc[:, 'ot dist'].apply(lambda x : float(np.array(x.getSample(1)))).to_dict()
    lower_bounds = list(df_input_factors[df_input_factors[column_to_fix] == False].loc[:, 'lower_bounds'].values)
    upper_bounds = list(df_input_factors[df_input_factors[column_to_fix] == False].loc[:, 'upper_bounds'].values)
    varying_input_factors = list(df_input_factors[df_input_factors[column_to_fix] == False].index)
    return dict_fixed_factors, lower_bounds, upper_bounds, varying_input_factors

dict_fixed_factors, lower_bounds, upper_bounds, varying_input_factors = fix_factors(df_input_factors, column_to_fix = 'fix_by_vi', fixed_value = 'mean')