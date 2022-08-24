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

with open(path_simulation_result, 'rb') as f:
    size = pickle.load(f)
    input_factor_names = pickle.load(f)
    distributions = pickle.load(f)
    static_params = pickle.load(f)


parameters = {}
for i in range(len(input_factor_names)):
    parameters[input_factor_names[i]] = [distributions[i].getA(), distributions[i].getB()]


with open('Studies/VARS/VARS_experiment.pkl', 'rb') as f: 
    experiment = pickle.load(f)
output = experiment.output
print(output.keys())

ivars_scale = 0.5 # Choose the scale range of interest, e.g., 0.1, 0.3, or 0.5

cols = list(parameters.keys()  )                   

# Rankings and Robustness
# print('Parameter Rankings'); 
# print(experiment.ivars_factor_ranking[cols])
# print('Robustness of Rankings')
# print(experiment.rel_ivars_factor_ranking[cols])

# Plot IVARS50, VARS-TO, and VARS-ABE along with their confidence intervals from Experiment 1

def show_factor_importance(experiment, logy = False):
    # factor importance bar chart for vars-abe, ivars50, and vars-to
    if 0.5 in experiment.ivars.index:
        # normalize data using mean normalization
        df1 = experiment.maee.unstack(0).iloc[0]
        df2 = experiment.st
        df3 = experiment.ivars.loc[0.5]

        normalized_maee = df1 / df1.sum()
        normalized_sobol = df2 / df2.sum()
        normalized_ivars50 = df3 / df3.sum()

        # plot bar chart
        x = np.arange(len(list(experiment.stub.columns)))  # the label locations
        width = 0.2  # the width of the bars

        barfig, barax = plt.subplots(figsize=(15,10))

        # if there are bootstrap results include them in bar chart
        if experiment.bootstrap_flag:
            # normalize confidence interval limits
            ivars50_err_upp = experiment.ivarsub.loc[0.5] / df3.sum()
            ivars50_err_low = experiment.ivarslb.loc[0.5] / df3.sum()
            sobol_err_upp = (experiment.stub / df2.to_numpy().sum()).to_numpy().flatten()
            sobol_err_low = (experiment.stlb / df2.to_numpy().sum()).to_numpy().flatten()
            maee_err_upp = experiment.maeeub.iloc[0] / df1.sum()
            maee_err_low = experiment.maeelb.iloc[0] / df1.sum()

            # subtract from normalized values so that error bars work properly
            ivars50_err_upp = np.abs(ivars50_err_upp - normalized_ivars50)
            ivars50_err_low = np.abs(ivars50_err_low - normalized_ivars50)
            sobol_err_upp = np.abs(sobol_err_upp - normalized_sobol)
            sobol_err_low = np.abs(sobol_err_low - normalized_sobol)
            maee_err_upp = np.abs(maee_err_upp - normalized_maee)
            maee_err_low = np.abs(maee_err_low - normalized_maee)

            # create error array for bar charts
            ivars50_err = np.array([ivars50_err_low, ivars50_err_upp])
            sobol_err = np.array([sobol_err_low, sobol_err_upp])
            maee_err = np.array([maee_err_low, maee_err_upp])

            rects1 = barax.bar(x - width, normalized_maee, width, label='VARS-ABE (Morris)', yerr=maee_err, color='green')
            rects2 = barax.bar(x, normalized_ivars50, width, label='IVARS50', yerr=ivars50_err, color='gold')
            rects3 = barax.bar(x + width, normalized_sobol, width, label='VARS-TO (Sobol)', yerr=sobol_err, color='lightblue')
        else:
            rects1 = barax.bar(x - width, normalized_maee, width, label='VARS-ABE (Morris)')
            rects2 = barax.bar(x, normalized_ivars50, width, label='IVARS50')
            rects3 = barax.bar(x + width, normalized_sobol, width, label='VARS-TO (Sobol)')

        # Add some text for labels, and custom x-axis tick labels, etc.
        barax.set_ylabel('Ratio of Factor Importance', fontsize=13)
        barax.set_xticks(x)
        barax.set_xticklabels(list(experiment.stub.columns))
        barax.legend(fontsize=11)

        barax.tick_params(labelrotation=90)

        barfig.tight_layout()

        if logy:
            barax.set_yscale('log')

        plt.show()


show_factor_importance(experiment)


#print(experiment.ivars50_grp[cols])


# fig_bar = plt.figure(figsize=(15,10))
# plt.gca().bar(cols, output['IVARS'].loc[pd.IndexSlice[ ivars_scale ]][cols], color='gold')
# plt.gca().set_title (r'Integrated variogram Across a Range of Scales', fontsize = 10)
# plt.gca().set_ylabel(r'IVARS-50 (Total-Variogram Effect)', fontsize = 13)
# plt.gca().set_xlabel(r'Model Parameter', fontsize=6)
# plt.gca().tick_params(labelrotation=90)
# plt.gca().grid()
# plt.gca().set_yscale('linear')
# plt.tight_layout()

# fig_bar = plt.figure(figsize=(15,10))
# plt.gca().bar(cols, output['IVARS'].loc[pd.IndexSlice[ ivars_scale ]][cols], color='gold')
# plt.gca().set_title (r'Integrated variogram Across a Range of Scales $[log-scale]$', fontsize = 10)
# plt.gca().set_ylabel(r'IVARS-50 (Total-Variogram Effect)', fontsize = 13)
# plt.gca().set_xlabel(r'Model Parameter', fontsize=6)
# plt.gca().tick_params(labelrotation=90)
# plt.gca().grid()
# plt.gca().set_yscale('log')
# plt.tight_layout()

# Plot VARS-TO 
# print("up")
# print(output['STub'].shape)
# print(output['STub'])
# print("low:")
# print(output['STlb'].shape)
# print(output['STlb'])
# print((output['STlb']).to_numpy().flatten())
# print("Output", "if list")
# exp = list(experiment.parameters.keys() )
# for i, col in enumerate(output['STlb'].columns):
#     print(col, cols[i], exp[i])

fig_bar = plt.figure(figsize=(15,10))

sobol_err_upp = np.abs((output['STub']).to_numpy().flatten() - output['ST'].to_frame().T.iloc[0])
sobol_err_low = np.abs((output['STlb']).to_numpy().flatten() - output['ST'].to_frame().T.iloc[0])
sobol_err = np.array([sobol_err_low, sobol_err_upp])

plt.gca().bar(output['STlb'].columns, output['ST'].to_frame().T.iloc[0],  yerr=sobol_err, color='lightblue')

plt.gca().set_title (r'Sobol Total-Order Effect', fontsize = 15)
plt.gca().set_ylabel(r'VARS-TO (Total-Order Effect)', fontsize = 13)
plt.gca().set_xlabel(r'Model Parameter', fontsize=6)
plt.gca().tick_params(labelrotation=90)
plt.gca().grid()
plt.gca().set_yscale('linear')
plt.tight_layout()

# fig_bar = plt.figure(figsize=(15,10))
# plt.gca().bar(cols, output['ST'].to_frame().T.iloc[0][cols], color='lightblue')
# plt.gca().set_title (r'Sobol Total-Order Effect $[log-scale]$', fontsize = 15)
# plt.gca().set_ylabel(r'VARS-TO (Total-Order Effect)', fontsize = 13)
# plt.gca().set_xlabel(r'Model Parameter', fontsize=6)
# plt.gca().tick_params(labelrotation=90)
# plt.gca().grid()
# plt.gca().set_yscale('log')
# plt.tight_layout()

# Plot VARS-ACE and  VARS-ABE
delta_of_interest = 0.1
out = output['MAEE'].to_frame().unstack(level=0).loc[delta_of_interest].to_frame().T.iloc[0]
fig_bar = plt.figure(figsize=(15,10))

maee_err_upp = np.abs((output['MAEEub'].iloc[0]).to_numpy().flatten() - out)
maee_err_low = np.abs((output['MAEElb'].iloc[0]).to_numpy().flatten() - out)
maee_err = np.array([maee_err_low, maee_err_upp])

plt.gca().bar(output['MAEEub'].columns, out, color='green', yerr = maee_err)
plt.gca().set_title (r'Mean Absolute Elementary Effect', fontsize = 15)
plt.gca().set_ylabel(r'VARS-ABE', fontsize = 13)
plt.gca().set_xlabel(r'Model Parameter', fontsize=6)
plt.gca().tick_params(labelrotation=90)
plt.gca().grid()
plt.gca().set_yscale('linear')
plt.tight_layout()

# fig_bar = plt.figure(figsize=(15,10))
# plt.gca().bar(cols, output['MEE'].to_frame().unstack(level=0).loc[delta_of_interest].to_frame().T.iloc[0], color='lightgreen')
# plt.gca().set_title (r'Mean Actual Elementary Effect ', fontsize = 15)
# plt.gca().set_ylabel(r'VARS-ACE', fontsize = 13)
# plt.gca().set_xlabel(r'Model Parameter', fontsize=6)
# plt.gca().tick_params(labelrotation=90)
# plt.gca().grid()
# plt.gca().set_yscale('linear')
# plt.tight_layout()

# Plot Directional Variograms 

# plotting_scale = 0.5 # any number between delta_h and one.

# #define the directional variogram
# variograms1 = output['Gamma'].unstack(0)[cols].copy()
# matrix_y = variograms1.loc[variograms1.index <= plotting_scale].to_numpy()
# column_x = variograms1.loc[variograms1.index <= plotting_scale].index.to_numpy()
# matrix_x = np.tile(column_x, (matrix_y.shape[1], 1)).T

# fig_cdf = plt.figure(figsize=(15,10))
# plt.gca().plot(matrix_x, matrix_y )
# plt.gca().set_title (r'Directional Variogram', fontsize = 15)
# plt.gca().set_ylabel(r'$γ(h)$', fontsize = 13)
# plt.gca().set_xlabel(r'$h$ (perturbation scale)', fontsize=13)
# plt.gca().set_yscale('linear')
# plt.gca().legend (cols, loc='upper left', fontsize = 10)
# plt.gca().grid()

# fig_cdf = plt.figure(figsize=(15,10))
# plt.gca().plot(matrix_x, matrix_y )
# plt.gca().set_title (r'Directional Variogram $[log-scale]$', fontsize = 15)
# plt.gca().set_ylabel(r'$γ(h)$', fontsize = 13)
# plt.gca().set_xlabel(r'$h$ (perturbation scale)', fontsize=13)
# plt.gca().set_yscale('log')
# plt.gca().legend (cols, loc='lower right', fontsize = 10)
# plt.gca().grid()

plt.show()
''' '''