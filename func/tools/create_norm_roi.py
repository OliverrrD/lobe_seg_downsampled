import nibabel as nib
import numpy as np
from scipy.ndimage.interpolation import zoom
import pandas as pd
from skimage import transform
import os

data_root = '/nfs/masi/gaor2/data/MCL/resample_roi'

save_root = '/nfs/masi/gaor2/data/MCL/roi_norm'

data_list = os.listdir(data_root)

for i in range(len(data_list)):
    print (i, data_list[i])
    img_nii = nib.load(data_root + '/' + data_list[i])
    img_new = img_nii.get_data()
    img_new[img_new < -1000] = -1000
    img_new[img_new > 1000] = 1000
    img_new = img_new.astype(np.int16)
    save_nii = nib.Nifti1Image(img_new, np.eye(4))
    nib.save(save_nii, save_root + '/' + data_list[i])