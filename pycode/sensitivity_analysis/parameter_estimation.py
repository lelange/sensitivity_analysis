import numpy as np
import pandas as pd
from datetime import datetime, date
from utils_SA import simulate_model
import pypesto
import pypesto.optimize as optimize
import pypesto.visualize as visualize
import pypesto.sample as sample
import pypesto.store as store
import matplotlib.pyplot as plt
import time
import json

from inputFactorSpace import input_factor_names, df_input_factors

path_data = 'data/worldometer_data.txt'

# Define Comartment names
compartments = ['susceptible', 'exposed', 'carrier', 'infected', 'hospitalized', 'icu', 'recovered', 'dead']
Compartments = ['Susceptible', 'Exposed', 'Carrier', 'Infected', 'Hospitalized', 'ICU', 'Recovered', 'Dead']
# Define age Groups
groups = ['0-4', '5-14', '15-34', '35-59', '60-79', '80+']
# Define population of age groups
populations = [40000, 70000, 190000, 290000, 180000, 60000] 

days = 50 # number of days to simulate
output_index = 3

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

# load data
data_dict = {}
with open(path_data) as f:
    lines = f.readlines()
    #data_dict.update(lines[0])
    print(lines[-1])
    for i in range(len(lines)-1):
        (key, value) = lines[i].split(":")
        value = json.loads(value)
        data_dict[key] = value
print(data_dict.keys())

start_dataset = 32
number_of_days = days
print(data_dict['categories'][start_dataset])
print(start_year, start_month, start_day)

# when used for a different time period, initial numbers must be matched first. 
# Saved simulation is from start day 0, therefore no initial numbers in Hospitalized, Infected and Recovered
# Divide by 100 because of the population size used in the model
#dataset = np.zeros((number_of_days+1, 2))
#dataset[:, 0] = np.array(data_dict['currently_infected'][start_dataset:number_of_days+1+start_dataset])/100.0
#dataset[:, 1] = np.array(data_dict['total_deaths'][start_dataset:number_of_days+1+start_dataset])/100.0

dead_cases = np.array(data_dict['total_deaths'][start_dataset:number_of_days+1+start_dataset])/100.0
infected_cases = np.array(data_dict['currently_infected'][start_dataset:number_of_days+1+start_dataset])/100.0

# load file for parameter importance and add to input factor dataframe
df_fixing = pd.read_csv('results/dgsm_vi_important.csv', index_col = 0)
df_input_factors['fix_by_vi']=[df_fixing.loc[name, 'vi'] for name in input_factor_names]
df_input_factors['use_all'] = [False for name in input_factor_names]
#df_input_factors = df_input_factors.set_index('Name', drop = True)

def fix_factors(df_input_factors, column_to_fix = 'fix_by_vi', fixed_value = 'mean'):
    dict_fixed_factors = {}
    if fixed_value == 'mean':
        dict_fixed_factors = df_input_factors[df_input_factors[column_to_fix]].loc[:, 'ot dist'].apply(lambda x : x.getMean()[0]).to_dict()
    if fixed_value == 'random':
        dict_fixed_factors = df_input_factors[df_input_factors[column_to_fix]].loc[:, 'ot dist'].apply(lambda x : float(np.array(x.getSample(1)))).to_dict()
    lower_bounds = list(df_input_factors[df_input_factors[column_to_fix] == False].loc[:, 'lower_bounds'].values)
    upper_bounds = list(df_input_factors[df_input_factors[column_to_fix] == False].loc[:, 'upper_bounds'].values)
    varying_input_factors = list(df_input_factors[df_input_factors[column_to_fix] == False].index)
    return dict_fixed_factors, lower_bounds, upper_bounds, varying_input_factors

dict_fixed_factors, lower_bounds, upper_bounds, varying_input_factors = fix_factors(df_input_factors, column_to_fix = 'use_all', fixed_value = 'mean')
# assumption that unknown cases are 3 times as much. Match infected numbers to observed.
#dict_fixed_factors['init_exposed'] = infected_cases[0]
#dict_fixed_factors['init_carrier'] = infected_cases[0]
#dict_fixed_factors['init_infected'] = infected_cases[0]

print(dict_fixed_factors)
print(len(input_factor_names) - len(varying_input_factors))
print(len(varying_input_factors))

def simulate_model_logParam(theta, 
                            param_names = varying_input_factors, 
                            static_params = static_params,
                            fixed_factors = dict_fixed_factors):
    '''
    Call the model simulation for given log-transformed 
    values (theta) for the parameters listed in param_names. 
    The static parameters include all other parameters that are not
    subject to the optimization.
    '''
    theta = np.exp(theta)
    if fixed_factors is None:
        result = simulate_model({**static_params, **dict(zip(param_names, theta))})
    else:
        result = simulate_model({**static_params, **dict(zip(param_names, theta)), **fixed_factors})
    return result


# TODO: change noise model
def neg_log_likelihood_logParam(theta, measurement = infected_cases):
    sigma = theta[0]
    simulation = simulate_model_logParam(theta[1:])
    simulation = simulation
    nllh = np.log(2*np.pi*sigma**2)+((measurement-simulation)/sigma)**2
    nllh = 0.5*sum(nllh)
    return nllh

def neg_binomial_log_likelihood():
    '''
    Discrete distribution that models nb of successes in a sequence of iid Bernoulli trials
    before r predefinded events (failures) occur. 
    Suitable if used to model the nb of infected/ pos. tested individuals as the event rate
    is known but the number of test is usually not.
    (Bernoulli trials: COVID tests, success: negative test, failure: positive test)
    r > 0 (int): number of failures until the experiment is stopped
    p in [0, 1] (float): success probability in each experiment
    '''
    return




#----------------------------------------------------------------------------------------------------------------------------------------------

objective1 = pypesto.Objective(
    fun=neg_log_likelihood_logParam,
)

# set upper and lower bound
#lb = np.array([1e-3] + [50, 25, 10, 10, 1e-3, 1e-3, 1e-3] + [3, 1, 1, 1, 1, 1] + [0.1, 0.6, 0.05, 0.01, 0.1, 0.15, 0.15, 1])
#ub = np.array([1] + [150, 75, 30, 30, 20, 20, 1] + [7, 15, 15, 15, 15, 15] + [ 0.9, 1.0, 0.5, 0.16, 0.35, 0.4, 0.77, 3])

lb = np.array([1e-5] + lower_bounds)
lb = np.where(lb <= 0, 1e-9, lb)
ub = np.array([1] + upper_bounds)#+1e-9 

lb = np.log(lb)
ub = np.log(ub)

problem1 = pypesto.Problem(objective=objective1, lb=lb, ub=ub)

# create different optimizers
optimizer_bfgs = optimize.ScipyOptimizer(method="l-bfgs-b")
optimizer_tnc = optimize.ScipyOptimizer()
optimizer_fides = optimize.FidesOptimizer()

# set number of starts
n_starts = 200
# save optimizer trace
history_options = pypesto.HistoryOptions(trace_record=True)

start = time.time()
# Run optimizaitons for different optimzers
# result1_fides = optimize.minimize(
#      problem=problem1,
#      optimizer=optimizer_fides,
#      n_starts=n_starts,
#      history_options=history_options,
#      filename=None,
#  )

result1_tnc = optimize.minimize(
    problem=problem1,
    optimizer=optimizer_tnc,
    n_starts=n_starts,
    history_options=history_options,
)

end = time.time()

sampler = sample.AdaptiveMetropolisSampler()
result = sample.sample(
    problem=problem1,
    sampler=sampler,
    n_samples=100000,
    result=result1_tnc,
)

fn = 'logs/'+ time.strftime("%Y-%m-%d %H%M%S") + '_optimizer_result_log.hdf5'

# Write result
store.write_result(
    result=result1_tnc,
    filename=fn,
    problem=True,
    optimize=True,
    sample=True,
)

print("It needed ", end - start, " seconds.")
#print("Fides")
#print((result1_fides.optimize_result.as_list()))
print("TNC")
print((result1_tnc.optimize_result.as_list()))
print("Data_____________________________")
print(result1_tnc.optimize_result.as_list()[0]['x'])
print(len(result1_tnc.optimize_result.as_list()[0]['x']))

# plot separated waterfalls

fig, ax1 = plt.subplots(1, 1, figsize=(15, 6))
# ax[0] = visualize.waterfall(result1_bfgs, size=(15, 6))
visualize.waterfall(result1_tnc, ax = ax1)
# ax[0].title.set_text('BFGS Waterfall')
ax1.title.set_text('TNC Waterfall')



# plot profiles
#pypesto.visualize.profiles(result1_tnc)

# plot samples
# pypesto.visualize.sampling_fval_traces(result1_tnc)

plt.show()

best_values = np.exp(result1_tnc.optimize_result.as_list()[0]['x'][1:])
print("best values: ", best_values)
static_params['output_index'] = [Compartments.index("Dead"), Compartments.index("Infected"), Compartments.index("Recovered")]
simulation = simulate_model({**static_params, **dict(zip(varying_input_factors, best_values)), **dict_fixed_factors})
print(simulation.shape)

datelist = np.array(pd.date_range(datetime(start_year, start_month,
                        start_day), periods=days, freq='D').strftime('%m-%d').tolist())

tick_range = (np.arange(int(days / 10) + 1) * 10)
tick_range[-1] -= 1
fig, ax = plt.subplots()
ax.plot(simulation[:, 0], 'o-', markersize = 3, label='simulation dead', color="darkorange") 
ax.plot(simulation[:, 1], 'o-', markersize = 3, label='simulation infected', color="darkgreen") 
ax.plot(simulation[:, 2], 'o-', markersize = 3, label='simulation recovered', color="navy") 
ax.plot(dead_cases, 'o-', markersize = 3, label='measurement dead', color = "bisque")
ax.plot(infected_cases, 'o-', markersize = 3, label='measurement infected', color = "limegreen")
#ax.plot(recovered_cases, 'o-', markersize = 3, label='measurement recovered', color = "royalblue")
ax.set_title("Simulated data and measured data")
ax.set_xticks(tick_range)
ax.set_xticklabels(datelist[tick_range], rotation=45)
ax.legend()
fig.tight_layout
plt.show()
fig.savefig('plots/'+ time.strftime("%Y-%m-%d %H%M%S") + '_Plot_best_simulation.pdf')
plt.close()
'''
'''
