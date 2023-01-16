'''Read freesurfer .stats files of subcortical and cortical structures'''

import pandas as pd

subcortical_file = open("C:/Users/joycerijs/freesurfer/subjects/bert/stats/aseg.stats")
cortex_rh_file = open("C:/Users/joycerijs/freesurfer/subjects/bert/stats/rh.aparc.DKTatlas.stats")
cortex_lh_file = open("C:/Users/joycerijs/freesurfer/subjects/bert/stats/lh.aparc.DKTatlas.stats")

# print(subcortical_file.readline(65))

# laatste line txt file printen:

# with open("C:/Users/joycerijs/freesurfer/subjects/bert/stats/aseg.stats") as f:
#     for line in f:
#         pass
#     last_line = line

# print(last_line)