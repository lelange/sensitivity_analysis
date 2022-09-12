# sensitivity_analysis

Repository contains python scripts and jupyter notebooks to conduct and analyse an uncertainty analysis 
of a SIR-type forecast model on the SARS-CoV-2 infection spread in Germany based on the model presented in [[1]](#1).

The project is based on the python package MEmilio which contains the analysed model: 
https://github.com/DLR-SC/memilio


## References
<a id="1">[1]</a> 
Kühn, M. J., Abele, D., Mitra, T., Koslow, W., Abedi, M., Rack, K., … Meyer-Hermann, M. (2021). 
Assessment of effective mitigation and prediction of the spread of SARS-CoV-2 in Germany using demographic information and spatial resolution. 
Mathematical Biosciences, 339, 108648. https://doi.org/10.1016/j.mbs.2021.108648


## File structure

- **utils_SA.py** contains the compuational model 
- **define(...).py** files contain scripts that prepare the data for the sensitivity analysis 
- **analysis(...).ipynb** files contain different types of sensitivty analyses from different libraries
- **parameter_estimation.py** contains a script to perform a parameter estimation on the whole input space or on a selected subset
- **inputFactorSpace.py** contains the definition of the input space (input factors and distributions)

