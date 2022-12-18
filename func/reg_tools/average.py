import numpy as np
import os
import nibabel as nib


data_root = '/nfs/masi/MCL/registration/affine_niftyreg/interp/ori'
data_list = os.listdir(data_root)

sum_img = np.zeros((620, 620, 600))

for i in range(10):
    print (i)
    nii = nib.load(data_root + '/' + data_list[i])
    img = nii.get_data()
    sum_img += img
    im_affine = nii.affine
    im_header = nii.header

ave_img = nib.Nifti1Image(sum_img / 10, affine=im_affine, header=im_header)
nib.save(ave_img, '/nfs/masi/MCL/registration/affine_niftyreg/average/ave.nii.gz')

