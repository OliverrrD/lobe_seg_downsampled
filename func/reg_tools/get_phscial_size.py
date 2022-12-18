import os
import nibabel as nib
import numpy as np
import pandas as pd

data_root = '/nfs/masi/MCL/nifti/combine'

save_csv_root = '/nfs/masi/MCL/xnat/xnat20200117/z_size.csv'

subj_list = os.listdir(data_root)

all_sess = []

all_size = []

for k in range(len(subj_list)):

    subj = subj_list[k]
    print (k , subj)
    if k > 100: break
    sess_list = os.listdir(data_root + '/' + subj)

    for i in range(len(sess_list)):
        img = nib.load(data_root + '/' + subj + '/' + sess_list[i] + '/' + subj + 'time' + sess_list[i] + '.nii.gz')
        header = img.header

        dims = header['dim'][1:4]
        pixdim = header['pixdim'][1:4]
        sizes = np.multiply(pixdim, dims)
        all_size.append(sizes[2])
        all_sess.append(subj + 'time' + sess_list[i])

data = pd.DataFrame()

data['session'] = all_sess
data['size'] = all_size

data.to_csv(save_csv_root, index = False)
