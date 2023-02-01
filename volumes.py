'''Read freesurfer .stats files of subcortical structures and create volume scatterplots with confidence ellipse'''

import pandas as pd
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


# Load file of new patient
file_path = "C:/Users/joycerijs/freesurfer/subjects/bert/stats/aseg.stats"
subcortical_file_new = open(file_path)
volume_dataframe_new = pd.read_table(subcortical_file_new, delim_whitespace=True, header=None, comment='#',
                                     usecols=[3, 4], names=[f'Volumes new', 'Name of structure'])
volume_index_new = volume_dataframe_new.set_index('Name of structure')
transposed_new = volume_index_new.T
string_icv = '# Measure EstimatedTotalIntraCranialVol'
with open(file_path, 'r') as txtfile:
    lines = txtfile.readlines()
    for line in lines:
        if line.find(string_icv) != -1:
            split = line.split(', ')
            icv_value = float(split[3])
new_volume_nor = transposed_new.div(icv_value, axis=0)
new_volume_norm = new_volume_nor.multiply(1000, axis=0)

# Path to normal database
path = "F:/Documenten/Universiteit/Master_TM+_commissies/Jaar 2/Stages/Stage 4/Bestanden voor project/map18 stats/volumestats/"
subcortical_files = [f for f in listdir(path) if isfile(join(path, f))]

# Create empty lists
values_list = []
names_list = []
icv_values = []
names_icv_values = []
structures = []

for i, j in enumerate(subcortical_files):
    volume_dataframe = pd.read_table(open(join(path, subcortical_files[i])), delim_whitespace=True, header=None,
                                     comment='#', usecols=[3, 4], names=[f'Volumes {str(j)[16:20]}',
                                     'Name of structure'])
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
    ax.scatter(x=volume_norm[f'Left-{structures_of_interest[k]}'], y=volume_norm[f'Right-{structures_of_interest[k]}'],
               alpha=0.5, label='Controls')
    ax.scatter(x=new_volume_norm[f'Left-{structures_of_interest[k]}'],
               y=new_volume_norm[f'Right-{structures_of_interest[k]}'], alpha=0.5, label='Current patient', marker='*')
    plt.title(f'{structures_of_interest[k]} Left vs. {structures_of_interest[k]} Right (ICV-normalized)')
    plt.xlabel(f'Volume left {structures_of_interest[k]} (mL)')
    plt.ylabel(f'Volume right {structures_of_interest[k]} (mL)')
    min_value = (volume_norm[[f'Left-{structures_of_interest[k]}',
                 f'Right-{structures_of_interest[k]}']].min(axis=1)).min(axis=0)
    max_value = (volume_norm[[f'Left-{structures_of_interest[k]}',
                 f'Right-{structures_of_interest[k]}']].max(axis=1)).max(axis=0)
    confidence_ellipse(volume_norm[f'Left-{structures_of_interest[k]}'],
                       volume_norm[f'Right-{structures_of_interest[k]}'], ax, n_std=2.0, label=r'$2\sigma$',
                       edgecolor='blue', linestyle=':')
    plt.legend()
    plt.xlim(min_value-0.25, max_value+0.25)
    plt.ylim(min_value-0.25, max_value+0.25)
    # plt.savefig(join(path, f'Left vs. Right-{structures_of_interest[k]} (ICV-normalized).jpg'))
    plt.show()
    plt.clf()
