'''Read freesurfer .stats files of subcortical and cortical structures'''

import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import scipy as sp
import scipy.stats as stats

# Load file of current patient
file_path = "C:/Users/joycerijs/freesurfer/subjects/bert/stats/aseg.stats"
subcortical_file_bert = open(file_path)
volume_dataframe_bert = pd.read_table(subcortical_file_bert, delim_whitespace=True, header=None, comment='#', usecols=[3, 4], names=[f'Volumes bert', 'Name of structure'])
volume_index_bert = volume_dataframe_bert.set_index('Name of structure')
transposed_bert = volume_index_bert.T
string_icv = '# Measure EstimatedTotalIntraCranialVol'
with open(file_path, 'r') as txtfile:
        lines = txtfile.readlines()
        for line in lines:
            if line.find(string_icv) != -1:
                split = line.split(', ')
                icv_value = float(split[3])
bert_volume_nor = transposed_bert.div(icv_value, axis=0)
bert_volume_norm = bert_volume_nor.multiply(1000, axis=0)

# Path to normal database
path = "F:/Documenten/Universiteit/Master_TM+_commissies/Jaar 2/Stages/Stage 4/Bestanden voor project/map18 stats/volumestats/"
subcortical_files = [f for f in listdir(path) if isfile(join(path, f))]

# print(subcortical_file.read())
values_list = []
names_list = []

icv_values = []
names_icv_values = []
structures = []

for i, j in enumerate(subcortical_files):
    volume_dataframe = pd.read_table(open(join(path, subcortical_files[i])), delim_whitespace=True, header=None, comment='#', usecols=[3, 4], names=[f'Volumes {str(j)[16:20]}', 'Name of structure'])
    structures = list(volume_dataframe['Name of structure'])
    names_list.append(f'Volumes {str(j)[16:20]}')
    values_list.append(list(volume_dataframe[f'Volumes {str(j)[16:20]}']))
    # Get ICV values
    with open(join(path, subcortical_files[i]), 'r') as txtfile:
        lines = txtfile.readlines()
        for line in lines:
            if line.find(string_icv) != -1:
                split = line.split(', ')
                icv_values.append(float(split[3]))

volume_df = pd.DataFrame(np.array(values_list), columns=structures, index=names_list)
volume_df['IntraCranialVolume'] = icv_values
volume_nor = volume_df.div(volume_df.IntraCranialVolume, axis=0)
volume_norm = volume_nor.multiply(1000, axis=0)

structures_of_interest = ['Hippocampus', 'Amygdala', 'Caudate', 'Thalamus-Proper', 'Putamen']
for k in range(len(structures_of_interest)):
    ax = plt.subplot()
    ax.set_aspect('equal')
    ax.scatter(x=volume_norm[f'Left-{structures_of_interest[k]}'], y=volume_norm[f'Right-{structures_of_interest[k]}'], alpha=0.5, label='Controls')
    ax.scatter(x=bert_volume_norm[f'Left-{structures_of_interest[k]}'], y=bert_volume_norm[f'Right-{structures_of_interest[k]}'], alpha=0.5, label='Current patient', marker='*')
    plt.title(f'{structures_of_interest[k]} Left vs. {structures_of_interest[k]} Right')
    plt.xlabel(f'Volume left {structures_of_interest[k]} (mL)')
    plt.ylabel(f'Volume right {structures_of_interest[k]} (mL)')
    min_value = (volume_norm[[f'Left-{structures_of_interest[k]}', f'Right-{structures_of_interest[k]}']].min(axis=1)).min(axis=0)
    max_value = (volume_norm[[f'Left-{structures_of_interest[k]}', f'Right-{structures_of_interest[k]}']].max(axis=1)).max(axis=0)
    plt.xlim(min_value-0.25, max_value+0.25)
    plt.ylim(min_value-0.25, max_value+0.25)
    # Linear line
    # plt.plot([min_value-0.25, max_value+0.25], [min_value-0.25, max_value+0.25], linestyle="--")
    # create linear regression
    a, b = np.polyfit(volume_norm[f'Left-{structures_of_interest[k]}'], volume_norm[f'Right-{structures_of_interest[k]}'], 1)
    xs = np.linspace(min_value-0.25, max_value+0.25)
    plt.plot(xs, a*xs+b, c='k', lw=2)
    ci = 0.95 * np.std(a*xs+b) / np.mean(a*xs+b) # Dit nog netter maken. volgens mij klopt het niet helemaal
    plt.fill_between(xs, (a*xs+b-ci), (a*xs+b+ci), color='blue', alpha=0.2)
    plt.legend()
    # plt.savefig(join(path, f'Left vs. Right-{structures_of_interest[k]} (ICV-normalized).jpg'))
    plt.show()
    plt.clf()

print('finish')
# # Van stack overflow, werkt nog niet: https://stackoverflow.com/questions/27164114/show-confidence-limits-and-prediction-limits-in-scatter-plot

# def plot_ci_manual(t, s_err, n, x, x2, y2, ax=None):
#     """Return an axes of confidence bands using a simple approach.
    
#     Notes
#     -----
#     .. math:: \left| \: \hat{\mu}_{y|x0} - \mu_{y|x0} \: \right| \; \leq \; T_{n-2}^{.975} \; \hat{\sigma} \; \sqrt{\frac{1}{n}+\frac{(x_0-\bar{x})^2}{\sum_{i=1}^n{(x_i-\bar{x})^2}}}
#     .. math:: \hat{\sigma} = \sqrt{\sum_{i=1}^n{\frac{(y_i-\hat{y})^2}{n-2}}}
    
#     References
#     ----------
#     .. [1] M. Duarte.  "Curve fitting," Jupyter Notebook.
#        http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/CurveFitting.ipynb
    
#     """
#     if ax is None:
#         ax = plt.gca()
    
#     ci = t * s_err * np.sqrt(1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
#     ax.fill_between(x2, y2 + ci, y2 - ci, color="#b9cfe7", edgecolor="")

#     return ax


# def plot_ci_bootstrap(xs, ys, resid, nboot=500, ax=None):
#     """Return an axes of confidence bands using a bootstrap approach.

#     Notes
#     -----
#     The bootstrap approach iteratively resampling residuals.
#     It plots `nboot` number of straight lines and outlines the shape of a band.
#     The density of overlapping lines indicates improved confidence.

#     Returns
#     -------
#     ax : axes
#         - Cluster of lines
#         - Upper and Lower bounds (high and low) (optional)  Note: sensitive to outliers

#     References
#     ----------
#     .. [1] J. Stults. "Visualizing Confidence Intervals", Various Consequences.
#        http://www.variousconsequences.com/2010/02/visualizing-confidence-intervals.html

#     """ 
#     if ax is None:
#         ax = plt.gca()

#     bootindex = sp.random.randint

#     for _ in range(nboot):
#         resamp_resid = resid[bootindex(0, len(resid) - 1, len(resid))]
#         # Make coeffs of for polys
#         pc = sp.polyfit(xs, ys + resamp_resid, 1)                   
#         # Plot bootstrap cluster
#         ax.plot(xs, sp.polyval(pc, xs), "b-", linewidth=2, alpha=3.0 / float(nboot))

#     return ax

# # Computations ----------------------------------------------------------------    
# # Modeling with Numpy
# def equation(a, b):
#     """Return a 1D polynomial."""
#     return np.polyval(a, b) 


# x = volume_norm['Left-Hippocampus']
# y = volume_norm['Right-Hippocampus']
# p, cov = np.polyfit(x, y, 1, cov=True)                     # parameters and covariance from of the fit of 1-D polynom.
# y_model = equation(p, x)                                   # model using the fit parameters; NOTE: parameters here are coefficients

# # Statistics
# n = volume_norm['Left-Hippocampus'].size                    # number of observations
# m = p.size                                                 # number of parameters
# dof = n - m                                                # degrees of freedom
# t = stats.t.ppf(0.975, n - m)                              # t-statistic; used for CI and PI bands

# # Estimates of Error in Data/Model
# resid = y - y_model                                        # residuals; diff. actual data from predicted values
# chi2 = np.sum((resid / y_model)**2)                        # chi-squared; estimates error in data
# chi2_red = chi2 / dof                                      # reduced chi-squared; measures goodness of fit
# s_err = np.sqrt(np.sum(resid**2) / dof)                    # standard deviation of the error

# # Plotting --------------------------------------------------------------------
# fig, ax = plt.subplots(figsize=(8, 6))

# # Data
# ax.plot(
#     x, y, "o", color="#b9cfe7", markersize=8, 
#     markeredgewidth=1, markeredgecolor="b", markerfacecolor="None"
# )

# # Fit
# ax.plot(x, y_model, "-", color="0.1", linewidth=1.5, alpha=0.5, label="Fit")  

# x2 = np.linspace(np.min(x), np.max(x), 100)
# y2 = equation(p, x2)

# # Confidence Interval (select one)
# plot_ci_manual(t, s_err, n, x, x2, y2, ax=ax)
# #plot_ci_bootstrap(x, y, resid, ax=ax)
   
# # Prediction Interval
# pi = t * s_err * np.sqrt(1 + 1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))   
# ax.fill_between(x2, y2 + pi, y2 - pi, color="None", linestyle="--")
# ax.plot(x2, y2 - pi, "--", color="0.5", label="95% Prediction Limits")
# ax.plot(x2, y2 + pi, "--", color="0.5")
# plt.show()


# # # test = (open(join(path, subcortical_files[0])))
# # # print(test.read())

# # # # comment = '#' geeft aan dat de lijnen met # overgeslagen worden.
# # # columns = volume_dataframe.columns.tolist()
# # # columns = columns[-1:] + columns[:-1]
# # # volume_dataframe = volume_dataframe[columns]

# # # left_structures = volume_dataframe[volume_dataframe['Name of structure'].str.contains('Left')]
# # # right_structures = volume_dataframe[volume_dataframe['Name of structure'].str.contains('Right')]

# # # print(volume_dataframe)
# # # # print(left_structures)
# # # # print(right_structures)
