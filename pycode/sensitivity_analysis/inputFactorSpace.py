import openturns as ot
from datetime import datetime, date

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