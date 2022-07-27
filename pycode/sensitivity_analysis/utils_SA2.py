from memilio.simulation import UncertainContactMatrix, ContactMatrix, Damping
from memilio.simulation.secir import SecirModel, simulate, AgeGroup, Index_InfectionState, SecirSimulation
from memilio.simulation.secir import InfectionState as State

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, date

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

def simulate_model2(
    infectious_mild_time = 6.,
    hospitalized_to_home_time = 12.,
    home_to_hospitalized_time = 5.,
    hospitalized_to_icu_time = 2.,
    icu_to_home_time = 8.,
    icu_to_death_time = 5.,
    #
    init_exposed = 100,
    init_carrier = 50, 
    init_infected = 20, 
    init_hospitalized = 20, 
    init_ICU = 10, 
    init_recovered = 10,
    init_dead = 0,
    #
    relative_carrier_infectability = 0.67, 
    infection_probability_from_contact = 1.0, 
    asymptotic_cases_per_infectious = 0.09, 
    risk_of_infection_from_symptomatic = 0.25,
    hospitalized_cases_per_infectious = 0.2, 
    ICU_cases_per_hospitalized = 0.25,
    deaths_per_hospitalized = 0.3, 
    max_risk_of_infection_from_symptomatic = 0.5,
    #
    num_groups = num_groups, 
    num_compartments = num_compartments,
    populations = populations,
    start_day = starting_day,
    baseline_contact_matrix = baseline_contact_matrix, 
    minimum_contact_matrix = minimum_contact_matrix,
    damping_coeff = 0.9, 
    days = days,
    dt = dt,
    # which compartment's maximal value should be outputed?
    output_index = compartments.index("Infected") ):

    '''
    TODO: Explain parameters
    # num_groups
    # num_compartments
    # start_date = (date(start_year, start_month, start_day) - date(start_year, 1, 1)).days
    # baseline_contact_matrix = baseline_contact_matrix, 
    # minimum_contact_matrix = minimum_contact_matrix
    # damping_coeff
    # damping_time
    # damping_level
    # daming_type
    # days
    # dt
    # output_index : index of compartment list where maximal value over time is given as output
    '''
    # Initialize Parameters
    model = SecirModel(num_groups)

    # Set parameters
    for i in range(num_groups):
        # Compartment transition duration
        model.parameters.IncubationTime[AgeGroup(i)] = 5.2
        model.parameters.InfectiousTimeMild[AgeGroup(i)] = infectious_mild_time
        model.parameters.SerialInterval[AgeGroup(i)] = 4.2
        model.parameters.HospitalizedToHomeTime[AgeGroup(i)] = hospitalized_to_home_time
        model.parameters.HomeToHospitalizedTime[AgeGroup(i)] = home_to_hospitalized_time
        model.parameters.HospitalizedToICUTime[AgeGroup(i)] = hospitalized_to_icu_time
        model.parameters.ICUToHomeTime[AgeGroup(i)] = icu_to_home_time
        model.parameters.ICUToDeathTime[AgeGroup(i)] = icu_to_death_time

        # Initial number of peaople in each compartment
        model.populations[AgeGroup(i), Index_InfectionState(State.Exposed)] = init_exposed
        model.populations[AgeGroup(i), Index_InfectionState(State.Carrier)] = init_carrier
        model.populations[AgeGroup(i), Index_InfectionState(State.Infected)] = init_infected
        model.populations[AgeGroup(i), Index_InfectionState(State.Hospitalized)] = init_hospitalized
        model.populations[AgeGroup(i), Index_InfectionState(State.ICU)] = init_ICU
        model.populations[AgeGroup(i), Index_InfectionState(State.Recovered)] = init_recovered
        model.populations[AgeGroup(i), Index_InfectionState(State.Dead)] = init_dead
        model.populations.set_difference_from_total((AgeGroup(i), Index_InfectionState(State.Susceptible)), populations[i])

         # Compartment transition propabilities
        model.parameters.RelativeCarrierInfectability[AgeGroup(i)] = relative_carrier_infectability  
        model.parameters.InfectionProbabilityFromContact[AgeGroup(i)] = infection_probability_from_contact
        model.parameters.AsymptoticCasesPerInfectious[AgeGroup(i)] = asymptotic_cases_per_infectious  # 0.01-0.16
        model.parameters.RiskOfInfectionFromSympomatic[AgeGroup(i)] = risk_of_infection_from_symptomatic  # 0.05-0.5
        model.parameters.HospitalizedCasesPerInfectious[AgeGroup(i)] = hospitalized_cases_per_infectious  # 0.1-0.35
        model.parameters.ICUCasesPerHospitalized[AgeGroup(i)] = ICU_cases_per_hospitalized  # 0.15-0.4
        model.parameters.DeathsPerHospitalized[AgeGroup(i)] = deaths_per_hospitalized  # 0.15-0.77
        model.parameters.MaxRiskOfInfectionFromSympomatic[AgeGroup(i)] = max_risk_of_infection_from_symptomatic * risk_of_infection_from_symptomatic # factor between 1 and 3

    model.parameters.StartDay = start_day

    # set contact rates and emulate some mitigations
    # set contact frequency matrix
    model.parameters.ContactPatterns.cont_freq_mat[0].baseline = baseline_contact_matrix
    model.parameters.ContactPatterns.cont_freq_mat[0].minimum = minimum_contact_matrix
    # Define Damping on Contacts
    model.parameters.ContactPatterns.cont_freq_mat.add_damping(Damping(coeffs = np.ones((num_groups, num_groups)) * damping_coeff, t = 30.0, level = 0, type = 0))
    
     # Apply mathematical constraints to parameters
    model.apply_constraints()

    # Run Simulation
    result = simulate(0, days, dt, model)
    
    # return maximal number of infected persons during the given time interval
    num_time_points = result.get_num_time_points()
    result_array = result.as_ndarray()
    t = result_array[0, :]
    group_data = np.transpose(result_array[1:, :])

    #sum over all groups
    data = np.zeros((num_time_points, num_compartments))
    for i in range(num_groups):
        data += group_data[:, i * num_compartments : (i + 1) * num_compartments]
    
    infections = data[:, output_index]
    return np.max(infections)