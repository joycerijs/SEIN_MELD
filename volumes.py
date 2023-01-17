'''Read freesurfer .stats files of subcortical and cortical structures'''

import pandas as pd

subcortical_file = open("C:/Users/joycerijs/freesurfer/subjects/bert/stats/aseg.stats")
cortex_rh_file = open("C:/Users/joycerijs/freesurfer/subjects/bert/stats/rh.aparc.DKTatlas.stats")
cortex_lh_file = open("C:/Users/joycerijs/freesurfer/subjects/bert/stats/lh.aparc.DKTatlas.stats")

# print(subcortical_file.read())

volume_dataframe = pd.read_table(subcortical_file, delim_whitespace=True, header=None, comment='#', usecols=[3, 4], names=['Volume in mm3', 'Name of structure'])
# comment = '#' geeft aan dat de lijnen met # overgeslagen worden.
columns = volume_dataframe.columns.tolist()
columns = columns[-1:] + columns[:-1]
volume_dataframe = volume_dataframe[columns]

left_structures = volume_dataframe[volume_dataframe['Name of structure'].str.contains('Left')]
right_structures = volume_dataframe[volume_dataframe['Name of structure'].str.contains('Right')]

# print(left_structures)
# print(right_structures)
