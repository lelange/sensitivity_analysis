# sensitivity_analysis

Repository contains python scripts and jupyter notebooks to conduct and analyse an uncertainty analysis 
of a SIR-type forecast model on the SARS-CoV-2 infection spread in Germany based on the model presented in [[1]](#1).

The project is based on the python package MEmilio which contains the analysed model: 
https://github.com/DLR-SC/memilio


## File structure

- **utils_SA.py** contains the compuational model 
- **define(...).py** files contain scripts that prepare the data for the sensitivity analysis 
- **analysis(...).ipynb** files contain different types of sensitivty analyses from different libraries
- **parameter_estimation.py** contains a script to perform a parameter estimation on the whole input space or on a selected subset
- **inputFactorSpace.py** contains the definition of the input space (input factors and distributions)


## References
<a id="1">[1]</a> 
Kühn, M. J., Abele, D., Mitra, T., Koslow, W., Abedi, M., Rack, K., … Meyer-Hermann, M. (2021). 
Assessment of effective mitigation and prediction of the spread of SARS-CoV-2 in Germany using demographic information and spatial resolution. 
Mathematical Biosciences, 339, 108648. https://doi.org/10.1016/j.mbs.2021.108648

<a id="1">[2]</a> 
Iwanaga, T., Usher, W., & Herman, J. (2022). Toward SALib 2.0: Advancing the accessibility and interpretability of global sensitivity analyses. Socio-Environmental Systems Modelling, 4, 18155. doi:10.18174/sesmo.18155

<a id="1">[3]</a> 
Herman, J. and Usher, W. (2017) SALib: An open-source Python library for sensitivity analysis. Journal of Open Source Software, 2(9). doi:10.21105/joss.00097

<a id="1">[4]</a>
Baudin, M., Dutfoy, A., Iooss, B., & Popelin, A.-L. (2017). OpenTURNS: An Industrial Software for Uncertainty Quantification in Simulation. In Handbook of Uncertainty Quantification (pp. 2001–2038). https://doi.org/10.1007/978-3-319-12385-1_64

<a id="1">[5]</a>
Razavi, S., Sheikholeslami, R., Gupta, H. V., & Haghnegahdar, A. (2019). VARS-TOOL: A toolbox for comprehensive, efficient, and robust sensitivity and uncertainty analysis. Environmental Modelling & Software, 112(May 2018), 95–107. https://doi.org/10.1016/j.envsoft.2018.10.005

<a id="1">[6]</a>
Stapor, P., Weindl, D., Ballnus, B., Hug, S., Loos, C., Fiedler, A., Krause, S., Hross, S., Fröhlich, F., Hasenauer, J. (2018). PESTO: Parameter EStimation TOolbox. Bioinformatics, 34(4), 705-707. doi: 10.1093/bioinformatics/btx676
https://github.com/ICB-DCM/pyPESTO

