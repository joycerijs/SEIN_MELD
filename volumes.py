'''Read freesurfer .stats files of subcortical and cortical structures'''

import pandas as pd
import os
from os import listdir
from os.path import isfile, join

# subcortical_file = open("C:/Users/joycerijs/freesurfer/subjects/bert/stats/aseg.stats")

path = "F:/Documenten/Universiteit/Master_TM+_commissies/Jaar 2/Stages/Stage 4/Bestanden voor project/map18 stats/volumestats/"
subcortical_files = [f for f in listdir(path) if isfile(join(path, f))]

# eerst proberen om een grafiekje te maken met 'Left-Hippocampus' en 'Right-Hippocampus'. Eigenlijk moet ik een grote dataframe hebben voor alle patiënten

# print(subcortical_file.read())
values_list = []
names_list = []
# nu even als dummy (f) toegevoegd. als ik dit in een loop doe kan ik alle volumes toevoegen van iedere patient in de dataframe.
for i, j in enumerate(subcortical_files):
    volume_dataframe = pd.read_table(open(join(path, subcortical_files[i])), delim_whitespace=True, header=None, comment='#', usecols=[3, 4], names=[f'Volumes {str(j)}', 'Name of structure'])
    # dit gaat mis. volume dataframe wordt telkens vervangen.
    names_list.append(f'Volumes {str(j)[16:20]}')
    values_list.append(list(volume_dataframe[f'Volumes {str(j)}']))
    # [16:20]
    # print(pd.read_table(open(join(path, subcortical_files[i])), delim_whitespace=True, header=None, comment='#', usecols=[3]))
    # volume_dataframe.insert(f'Volumes {str(j)[16:20]}', pd.read_table(open(join(path, subcortical_files[i])), delim_whitespace=True, header=None, comment='#', usecols=[3]))

all_volumes_dataframe = pd.DataFrame(data = values_list, columns=names_list)
# ValueError: 39 columns passed, passed data had 45 columns

print(names_list)
print(values_list)

# test = (open(join(path, subcortical_files[0])))
# print(test.read())

# # comment = '#' geeft aan dat de lijnen met # overgeslagen worden.
# columns = volume_dataframe.columns.tolist()
# columns = columns[-1:] + columns[:-1]
# volume_dataframe = volume_dataframe[columns]

# left_structures = volume_dataframe[volume_dataframe['Name of structure'].str.contains('Left')]
# right_structures = volume_dataframe[volume_dataframe['Name of structure'].str.contains('Right')]

# print(volume_dataframe)
# # print(left_structures)
# # print(right_structures)
