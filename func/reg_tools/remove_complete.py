
import os

inter_root = "/nfs/masi/MCL/registration/affine_niftyreg/MCL_reg_20200120/interp"

complet_root = '/nfs/masi/MCL/registration/affine_niftyreg/MCL_reg_20200120/complete/interpolation'


mask_list = os.listdir(inter_root + '/mask')

ori_list = os.listdir(inter_root + '/ori')

complet_list = os.listdir(complet_root)

true_complet_list = list (set(mask_list) & set(ori_list))

print (len(true_complet_list))

for i in range(len(complet_list)):
    if complet_list[i] not in true_complet_list:
        print ('rm ' + complet_root + '/' + complet_list[i])
        os.system('rm ' + complet_root + '/' + complet_list[i])