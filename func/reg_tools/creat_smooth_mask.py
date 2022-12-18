import nibabel as nib
import scipy.ndimage
import numpy as np

ori_mask_path = '/nfs/masi/xuk9/NLST/registration/affine_niftyreg/20191205_NLST_non_cancer_reg_affine_ori_mask_local/interp/common_region_map_90.nii.gz'

new_mask_path = '/nfs/masi/MCL/registration/affine_niftyreg_0/region_mask/di_er_8.nii.gz'

nii = nib.load(ori_mask_path)

img = nii.get_data()

tmp1 = scipy.ndimage.morphology.binary_dilation(img, structure = np.ones((8, 8, 8)), iterations = 3)

tmp2 = scipy.ndimage.morphology.binary_erosion(tmp1,structure = np.ones((8, 8, 8)), iterations = 3)

tmp2 = tmp2.astype('float')

nii_mask = nib.Nifti1Image(tmp2, affine=nii.affine)
nib.save(nii_mask, new_mask_path)

