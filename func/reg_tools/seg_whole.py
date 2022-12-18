import os
import numpy as np
import nibabel as nib

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_roi_z_clip_only(mask_path):

    mask_nii = nib.load(mask_path)
    mask = mask_nii.get_data()

    if mask.shape[2] * mask_nii.affine[2][2] < 450:
       return None

    z_list = []
    for i in range(mask.shape[2]):
        if np.sum(mask[:, :, i]) > 20:
            z_list.append(i)

    z_begin, z_end = z_list[0], z_list[-1]

    z_dim = z_end - z_begin

    z_begin = max(0, int(z_begin - z_dim * 0.07))

    z_dim = int(z_dim * 1.15)

    x_begin = 0
    x_dim = mask.shape[0]
    y_begin = 0
    y_dim = mask.shape[1]


    return x_begin, x_dim, y_begin, y_dim, z_begin, z_dim

data_root = '/nfs/masi/MCL/wholebody_nifti/combine'
mask_root = '/nfs/masi/MCL/wholebody_nifti/mask'
new_root = '/nfs/masi/MCL/wholebody_nifti/crop'

subj_list = os.listdir(data_root)

for i in range(len(subj_list)):
    print (i, len(subj_list))
    #if i > 10: break
    sess_list = os.listdir(data_root + '/' + subj_list[i])
    for j in range(len(sess_list)):
        item = os.listdir(data_root + '/' + subj_list[i] + '/' + sess_list[j])[0]
        makedir(new_root + '/' + subj_list[i] + '/' + sess_list[j])
        index = get_roi_z_clip_only(mask_root + '/' + subj_list[i] + '/' + sess_list[j] + '/' + item)
        ori_path = data_root + '/' + subj_list[i] + '/' + sess_list[j] + '/' + item
        save_path = new_root + '/' + subj_list[i] + '/' + sess_list[j] + '/' + item
        if index == None:
            print ('----cp '  +  ori_path + ' ' + save_path)
            os.system('cp '  +  ori_path + ' ' + save_path)
        else:
            print ('fslroi ' +  ori_path + ' ' + save_path + ' ' + str(index[0]) + ' ' + str(index[1]) + ' ' + str(index[2])
               + ' ' + str(index[3])+ ' ' + str(index[4]) + ' ' + str(index[5]))
            os.system('fslroi ' +  ori_path + ' ' + save_path + ' ' + str(index[0]) + ' ' + str(index[1]) + ' ' + str(index[2])
               + ' ' + str(index[3])+ ' ' + str(index[4]) + ' ' + str(index[5]))



#
# mask_roots = ['/nfs/masi/MCL/registration/affine_niftyreg_0/temp/lung_mask',
#               '/nfs/masi/MCL/registration/affine_niftyreg/MCL_reg_20200115/temp/lung_mask',
#               '/nfs/masi/MCL/registration/affine_niftyreg/MCL_reg_20200120/temp/lung_mask']
#
# mask0 = os.listdir(mask_roots[0])
# mask1 = os.listdir(mask_roots[1])
# mask2 = os.listdir(mask_roots[2])
#
# subj_list = os.listdir(data_root)
#
# for i in range(len(subj_list)):
#     print (i, len(subj_list))
#     sess_list = os.listdir(data_root + '/' + subj_list[i])
#     for j in range(len(sess_list)):
#         item = os.listdir(data_root + '/' + subj_list[i] + '/' + sess_list[j])[0]
#         makedir(new_root + '/' + subj_list[i] + '/' + sess_list[j])
#         if item in mask0:
#             os.system('cp ' + mask_roots[0] + '/' + item + ' ' + new_root + '/' + subj_list[i] + '/' + sess_list[j])
#         if item in mask1:
#             os.system('cp ' + mask_roots[1] + '/' + item + ' ' + new_root + '/' + subj_list[i] + '/' + sess_list[j])
#         if item in mask2:
#             os.system('cp ' + mask_roots[2] + '/' + item + ' ' + new_root + '/' + subj_list[i] + '/' + sess_list[j])