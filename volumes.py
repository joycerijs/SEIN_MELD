'''Read freesurfer .stats files of subcortical and cortical structures'''

subcortical_file = open("C:/Users/joycerijs/freesurfer/subjects/bert/stats/aseg.stats")
cortex_rh_file = open("C:/Users/joycerijs/freesurfer/subjects/bert/stats/rh.aparc.DKTatlas.stats")
cortex_lh_file = open("C:/Users/joycerijs/freesurfer/subjects/bert/stats/lh.aparc.DKTatlas.stats")

print(subcortical_file.read())