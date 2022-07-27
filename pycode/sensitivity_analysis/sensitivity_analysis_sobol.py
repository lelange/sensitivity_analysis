import memilio.simulation.secir as secir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime, date
import time
import seaborn as sns
plt.style.use("seaborn")
from utils_SA import simulate_model, generate_output_daywise

# openturns libraries
#from __future__ import print_function
import openturns as ot
import openturns.viewer as viewer
from matplotlib import pylab as plt
ot.Log.Show(ot.Log.NONE)

# TODO: move to utils
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

# set contact frequency matrix
baseline_contact_matrix0 = "../../data/contacts/baseline_home.txt"
baseline_contact_matrix1 = "../../data/contacts/baseline_school_pf_eig.txt"
baseline_contact_matrix2 = "../../data/contacts/baseline_work.txt"
baseline_contact_matrix3 = "../../data/contacts/baseline_other.txt"

baseline_contact_matrix = np.loadtxt(baseline_contact_matrix0) \
        + np.loadtxt(baseline_contact_matrix1) \
        + np.loadtxt(baseline_contact_matrix2) + np.loadtxt(baseline_contact_matrix3)

minimum_contact_matrix = np.ones((num_groups, num_groups)) * 0

static_params = {
    'num_groups': num_groups, 
    'num_compartments': num_compartments,
    'populations': populations,
    'start_day' : (date(start_year, start_month, start_day) - date(start_year, 1, 1)).days,
    'baseline_contact_matrix' : baseline_contact_matrix, 
    'minimum_contact_matrix' : minimum_contact_matrix,
    'damping_coeff' : 0.9, 
    'days' : days,
    'dt' : dt,
    # which compartment's value should be outputed?
    'output_index' : [compartments.index("Infected")]
}

# define input factors and their distributions
# TODO define "incubation_time" and "serial_interval" later
# todo max_rist_of_infection_from symptomatic -> twice rist_of_infection...

comp_transition_duration = [ 
    "infectious_mild_time",   
    "hospitalized_to_home_time", 
    "home_to_hospitalized_time",
    "hospitalized_to_icu_time", 
    "icu_to_home_time", 
    "icu_to_death_time"
]
# TODO does this make sense? Intial values should just depend on the time.

initial_numbers_comp = [
    "init_exposed", 
    "init_carrier", 
    "init_infected", 
    "init_hospitalized", 
    "init_ICU", 
    "init_recovered",
    "init_dead"
]

comp_transition_probabilities = [
    "relative_carrier_infectability", 
    "infection_probability_from_contact", 
    "asymptotic_cases_per_infectious", 
    "risk_of_infection_from_symptomatic",
    "hospitalized_cases_per_infectious", 
    "ICU_cases_per_hospitalized",
    "deaths_per_hospitalized", 
    "max_risk_of_infection_from_symptomatic"
]


dist_comp_transition_duration = [ 
    ot.Uniform(3, 7),
    ot.Uniform(1, 15),
    ot.Uniform(1, 15),
    ot.Uniform(1, 15),
    ot.Uniform(1, 15),
    ot.Uniform(1, 15)
]
"""
ot.Uniform(3, 7),
ot.Uniform(5, 14),
ot.Uniform(3, 11),
ot.Uniform(1, 7),
ot.Uniform(5, 10),
ot.Uniform(1, 7)
"""
dist_initial_numbers_comp = [
    ot.Uniform(50, 150),
    ot.Uniform(25, 75),
    ot.Uniform(10, 30),
    ot.Uniform(10, 30),
    ot.Uniform(0, 20),
    ot.Uniform(0, 20),
    ot.Uniform(0, 1)
]

dist_comp_transition_probabilities = [
    ot.Uniform(0.1, 0.9),
    ot.Uniform(0.6, 1.0),
    ot.Uniform(0.05, 0.5),
    ot.Uniform(0.01, 0.16),
    ot.Uniform(0.1, 0.35),
    ot.Uniform(0.15, 0.4),
    ot.Uniform(0.15, 0.77),
    ot.Uniform(1, 3) #factor by which risk_of_infection_from_symptomatic gets multiplied
]


input_factor_names = comp_transition_duration + initial_numbers_comp + comp_transition_probabilities
coll = dist_comp_transition_duration + dist_initial_numbers_comp + dist_comp_transition_probabilities

dimension = len(input_factor_names)

inputDistribution = ot.ComposedDistribution(coll)
inputDistribution.setDescription(input_factor_names)

size = 1000 

sie = ot.SobolIndicesExperiment(inputDistribution, size)

# generate samples from the input distribution
inputDesign = sie.generate()
input_names = inputDistribution.getDescription()
inputDesign.setDescription(input_factor_names)

#print("Sample size: ", inputDesign.getSize())
static_params["output_operation"]="max"
static_params["output_index"] = [compartments.index("Dead")]

def dict_generate_output_daywise(inputDesign, input_factor_names = input_factor_names, static_params = static_params):
    # how many timepoints does the integration return?
    output = np.zeros((len(inputDesign), static_params["days"]+1))
    
    for i in range(len(inputDesign)):
        result = simulate_model({**dict(zip(input_factor_names, inputDesign[i])), **static_params})
        output[i] = result.T
        
    return output

start = time.time()
output = generate_output_daywise(inputDesign, input_factor_names, static_params)
outputDesign = ot.Sample(output)
np.savetxt("simulation_output.txt", output)
np.savetxt("inputDesign.txt", )
end = time.time()
print(end - start)

sensitivityAnalysis = ot.SaltelliSensitivityAlgorithm(inputDesign, outputDesign, size)

agg_first_order = sensitivityAnalysis.getAggregatedFirstOrderIndices()
agg_total_order = sensitivityAnalysis.getAggregatedTotalOrderIndices()
print("Agg. first order indices: ", agg_first_order)
print("Agg. total order indices: ", agg_total_order)

graph = sensitivityAnalysis.draw()
view = viewer.View(graph, (1400, 500))
view.save('sobol.png', dpi=100)

view.show()