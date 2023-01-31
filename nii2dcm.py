'''Convert nii to DICOM and change metadata'''

import nibabel
import pydicom
import numpy as np
import os
from tqdm import tqdm
import glob
import difflib
import datetime


def convertNsave(type, arr, dicom_dir, file_dir, now_series, now_study, index=0):
    """
    'type': Parameter defines if it is the T1 or FLAIR prediction
    'arr': Parameter will take a numpy array that represents only one slice.
    'file_dir': Parameter will take the path to save the slices
    'index': Parameter will represent the index of the slice, so this parameter will be used to put
    the name of each slice while using a for loop to convert all the slices
    'dicom dir': Parameter defines directory of original dicom file
    """

    dicom_file = pydicom.dcmread(dicom_dir)
    arr = arr.astype('uint16')
    dicom_file.Rows = arr.shape[0]
    dicom_file.Columns = arr.shape[1]
    dicom_file.PhotometricInterpretation = "MONOCHROME2"
    dicom_file.SamplesPerPixel = 1
    dicom_file.BitsStored = 16
    dicom_file.BitsAllocated = 16
    dicom_file.HighBit = 15
    dicom_file.PixelRepresentation = 1
    SeriesInstanceUID = dicom_file.SeriesInstanceUID[0:40]
    if type == 'T1':
        dicom_file.ImageType = "Derived/secondary/MELD_T1_prediction"
        dicom_file.SeriesDescription = 'MELD_T1_prediction'
        dicom_file.ProtocolName = 'MELD_T1_prediction'
        dicom_file.SeriesInstanceUID = SeriesInstanceUID + now_series
        dicom_file.SOPInstanceUID = dicom_file.SeriesInstanceUID[0:53] + (str(int(dicom_file.SeriesInstanceUID[53])+1))
        dicom_file.file_meta.MediaStorageSOPInstanceUID = dicom_file.SOPInstanceUID
        dicom_file.StudyInstanceUID = SeriesInstanceUID + now_study  # Datum is nu op andere volgorde. De eerste getallen van seriesinstanceUID zijn hetzelfde als van study
    if type == 'FLAIR':
        dicom_file.ImageType = "Derived/secondary/MELD_FLAIR_prediction"
        dicom_file.SeriesDescription = 'MELD_FLAIR_prediction'
        dicom_file.ProtocolName = 'MELD_FLAIR_prediction'
        dicom_file.SeriesInstanceUID = SeriesInstanceUID + now_series + '1'
        dicom_file.SOPInstanceUID = dicom_file.SeriesInstanceUID[0:53] + (str(int(dicom_file.SeriesInstanceUID[53])+1)) + '1'
        dicom_file.file_meta.MediaStorageSOPInstanceUID = dicom_file.SOPInstanceUID
        dicom_file.StudyInstanceUID = SeriesInstanceUID + now_study + '1'
    dicom_file.PixelData = arr.tobytes()
    # dicom_file.save_as(os.path.join(file_dir, f'slice{index+1}.dcm'))


def nifti2dicom(type, nifti_dir, dicom_dir, out_dir):
    """
    This function is to convert one nifti file into dicom series
    'type': Defines if it is the T1 or FLAIR prediction
    'nifti_dir': Parameter defines the path to the nifti file
    'out_dir': Parameter defines the path to output
    'dicom dir': Parameter defines directory of original dicom file
    """

    nifti_file = nibabel.load(nifti_dir)
    nifti_array_ = nifti_file.get_fdata()
    nifti_array = np.rot90(np.fliplr(nifti_array_), 1)
    number_slices = nifti_array.shape[2]
    now_series = datetime.datetime.now().strftime('%Y%m%d%H%M%S') # De actuele tijd wordt 1x berekend en voor alle slices gebruikt
    now_study = datetime.datetime.now().strftime('%S%M%H%d%m%Y')
    for slice_ in tqdm(range(number_slices)):
        convertNsave(type, nifti_array[:, :, slice_], dicom_dir[slice_], out_dir, now_series, now_study, slice_)

# Define directories
nifti_dir_T1 = 'f:/Documenten/Universiteit/Master_TM+_commissies/Jaar 2/Stages/Stage 4/Bestanden voor project/prediction-in-T1-dcm.nii'
nifti_dir_FLAIR = 'f:/Documenten/Universiteit/Master_TM+_commissies/Jaar 2/Stages/Stage 4/Bestanden voor project/prediction-in-FLAIR-dcm.nii'
out_dir_T1 = 'f:/Documenten/Universiteit/Master_TM+_commissies/Jaar 2/Stages/Stage 4/Bestanden voor project/DCM prediction T1'
out_dir_FLAIR = 'f:/Documenten/Universiteit/Master_TM+_commissies/Jaar 2/Stages/Stage 4/Bestanden voor project/DCM prediction FLAIR'
dicom_dir_T1 = glob.glob("f:/Documenten/Universiteit/Master_TM+_commissies/Jaar 2/Stages/Stage 4/Bestanden voor project/rawDCM/Mri Hersenen Epilepsie/s T1 3D HR - 801/*.dcm")
dicom_dir_FLAIR = glob.glob("f:/Documenten/Universiteit/Master_TM+_commissies/Jaar 2/Stages/Stage 4/Bestanden voor project/rawDCM/Mri Hersenen Epilepsie/t T2 3D TSE FLAIR - 701/*.dcm")

# Convert the files
nifti2dicom('T1', nifti_dir_T1, dicom_dir_T1, out_dir_T1)
nifti2dicom('FLAIR', nifti_dir_FLAIR, dicom_dir_FLAIR, out_dir_FLAIR)

# Compare DICOM metadata

T1_map = 'f:/Documenten/Universiteit/Master_TM+_commissies/Jaar 2/Stages/Stage 4/Bestanden voor project/map18/map18_combined_z_score - 18003/IM-0001-0001-0001.dcm'
T1_prediction = 'f:/Documenten/Universiteit/Master_TM+_commissies/Jaar 2/Stages/Stage 4/Bestanden voor project/DCM prediction T1/slice1.dcm'

datasets = tuple([pydicom.dcmread(filename, force=True)
                  for filename in (T1_map, T1_prediction)])

# difflib compare functions require a list of lines, each terminated with
# newline character massage the string representation of each dicom dataset
# into this form:
rep = []
for dataset in datasets:
    lines = str(dataset).split("\n")
    lines = [line + "\n" for line in lines]  # add the newline to end
    rep.append(lines)

diff = difflib.Differ()
for line in diff.compare(rep[0], rep[1]):
    if line[0] != "?":
        print(line)