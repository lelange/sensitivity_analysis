import numpy as np
import pandas as pd
from datetime import datetime, date
from utils_SA import simulate_model
import pypesto
import pypesto.optimize as optimize
import pypesto.visualize as visualize
import matplotlib.pyplot as plt
import time

# Define parameter name, inital values (log transfomed) and boundary values

initial_numbers_comp = [
    "init_exposed", 
    "init_carrier", 
    "init_infected", 
    "init_hospitalized", 
    "init_ICU", 
    "init_recovered",
    "init_dead"
]

comp_transition_duration = [ 
    "infectious_mild_time",   
    "hospitalized_to_home_time", 
    "home_to_hospitalized_time",
    "hospitalized_to_icu_time", 
    "icu_to_home_time", 
    "icu_to_death_time"
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

parameter_names = initial_numbers_comp + comp_transition_duration + comp_transition_probabilities
# Define Comartment names
compartments = ['Susceptible', 'Exposed', 'Carrier', 'Infected', 'Hospitalized', 'ICU', 'Recovered', 'Dead']
# Define age Groups
groups = ['0-4', '5-14', '15-34', '35-59', '60-79', '80+']
# Define population of age groups
populations = [40000, 70000, 190000, 290000, 180000, 60000] 

days = 50 # number of days to simulate
start_day = 1
start_month = 3
start_year = 2020
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
    'output_index' : [compartments.index("Infected")],
    'output_operation' : None
}

# load measurement data
df=pd.read_csv('death_confirmed_recovered.csv', sep=',')
df = (df.T).set_axis(['dead', 'confirmed', 'recovered'], axis=1, inplace=False)

date = str(start_month)+'/'+str(start_day)+'/'+ str(start_year)[2:]
date_index = df.index.get_loc(date)

dead_cases = np.squeeze(df["dead"].iloc[date_index:date_index+days+1].values)
confirmed_cases = np.squeeze(df["confirmed"].iloc[date_index:date_index+days+1].values)
recovered_cases = np.squeeze(df["recovered"].iloc[date_index:date_index+days+1].values)

dead_cases = dead_cases + np.ones(len(dead_cases))*1e-5


def simulate_model_logParam(theta, param_names = parameter_names, static_params = static_params):
    '''
    Call the model simulation for given log-transformed 
    values (theta) for the parameters listed in param_names. 
    The static parameters include all other parameters that are not
    subject to the optimization.
    '''
    theta = np.exp(theta)
    result = simulate_model({**static_params, **dict(zip(parameter_names, theta))})
    return result


# TODO: change noise model
def neg_log_likelihood_logParam(theta, measurement = confirmed_cases):
    sigma = theta[0]
    simulation = simulate_model_logParam(theta[1:])
    simulation = np.cumsum(simulation)
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

lb = np.array([1e-5] + [50, 25, 10, 10, 1e-5, 1e-5, 1e-5] + [3, 1, 1, 1, 1, 1] + [0.1, 0.6, 0.05, 0.01, 0.1, 0.15, 0.15, 1])
ub = np.array([1] + [150, 75, 30, 30, 20, 20, 1] + [7, 15, 15, 15, 15, 15] + [ 0.9, 1.0, 0.5, 0.16, 0.35, 0.4, 0.77, 3])

lb = np.log(lb)
ub = np.log(ub)

problem1 = pypesto.Problem(objective=objective1, lb=lb, ub=ub)

# create different optimizers
optimizer_bfgs = optimize.ScipyOptimizer(method="l-bfgs-b")
optimizer_tnc = optimize.ScipyOptimizer(method="TNC")

# set number of starts
n_starts = 15
# save optimizer trace
history_options = pypesto.HistoryOptions(trace_record=True)

start = time.time()
# Run optimizaitons for different optimzers
# result1_bfgs = optimize.minimize(
#     problem=problem1,
#     optimizer=optimizer_bfgs,
#     n_starts=n_starts,
#     history_options=history_options,
#     filename=None,
# )

result1_tnc = optimize.minimize(
    problem=problem1,
    optimizer=optimizer_tnc,
    n_starts=n_starts,
    history_options=history_options,
    filename='logs/'+ time.strftime("%Y-%m-%d %H%M%S") + '_optimizer_log.hdf5',
)

end = time.time()
print("It needed ", end - start, " seconds.")
#print("BFGS")
#print((result1_bfgs.optimize_result.as_list()))
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
static_params['output_index'] = [compartments.index("Dead"), compartments.index("Infected"), compartments.index("Recovered")]
simulation = simulate_model({**static_params, **dict(zip(parameter_names, best_values))})
print(simulation.shape)

datelist = np.array(pd.date_range(datetime(start_year, start_month,
                        start_day), periods=days, freq='D').strftime('%m-%d').tolist())

tick_range = (np.arange(int(days / 10) + 1) * 10)
tick_range[-1] -= 1
fig, ax = plt.subplots()
ax.plot(simulation[:, 0], 'o-', markersize = 3, label='simulation dead', color="darkorange") 
ax.plot(np.cumsum(simulation[:, 1]), 'o-', markersize = 3, label='simulation infected (cumsum)', color="darkgreen") 
ax.plot(simulation[:, 2], 'o-', markersize = 3, label='simulation recovered', color="navy") 
ax.plot(dead_cases, 'o-', markersize = 3, label='measurement dead', color = "bisque")
ax.plot(confirmed_cases, 'o-', markersize = 3, label='measurement infected', color = "limegreen")
ax.plot(recovered_cases, 'o-', markersize = 3, label='measurement recovered', color = "royalblue")
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


