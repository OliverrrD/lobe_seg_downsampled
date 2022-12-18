import os

ori_root = '/nfs/masi/xuk9/NLST/registration/affine_niftyreg/20200201_NLST_missing_data_local_config_v4/interp/ori'
mask_root = '/nfs/masi/xuk9/NLST/registration/affine_niftyreg/20200201_NLST_missing_data_local_config_v4/interp/mask'

slurm_root = '/nfs/masi/NLST/registration/20200201_NLST_missing_data_local_config_v4/After_REG/slurm/script'
log_root = '/nfs/masi/NLST/registration/20200201_NLST_missing_data_local_config_v4/After_REG/slurm/log'

save_nii_root = '/nfs/masi/NLST/registration/20191205_NLST_non_cancer_reg_affine_ori_mask_local/After_REG/reg_crop'
save_npy_root = '/nfs/masi/NLST/registration/20191205_NLST_non_cancer_reg_affine_ori_mask_local/After_REG/REG_DSB/prep'

xb, xe, yb, ye, zb, ze = '95', '525', '166', '454', '97', '503'

    #80, 460, 145, 399, 110, 435,     # 20200115 for 1x1x1


    # '80', '460', '145', '399', '110', '435' # this is the actually use for 1x1x1 image, new pipeline

ori_list = os.listdir(ori_root)


# for i in range(len(ori_list)):
#
#     #if i > 1: break
#     save_npy = save_npy_root + '/' + ori_list[i].replace('.nii.gz', '_clean.npy')
#     save_nii = save_nii_root + '/' + ori_list[i]
#
#     f = open(slurm_root + '/' + ori_list[i].replace('.nii.gz', '.sh'), 'w')
#
#
#     f.write('#!/bin/bash\n')
#     f.write('#SBATCH --job-name=' + ori_list[i]  + '\n')
#     f.write('#SBATCH --output=' + log_root + '/' + ori_list[i].replace('.nii.gz', '.txt') + '\n')
#     f.write('#SBATCH --ntasks=1\n')
#     f.write('#SBATCH --time=07:00:00\n')
#     f.write('#SBATCH --mem-per-cpu=11G\n')
#
#
#     f.write('/home/local/VANDERBILT/gaor2/anaconda3/envs/python36/bin/python /home/local/VANDERBILT/gaor2/code/ThoraxRegistration/rg_tools/convert_high2low.py '
#             + '--ori_path ' + ori_root + '/' + ori_list[i] + ' --mask_path ' + mask_root + '/'
#             + ori_list[i] + ' --xb ' + xb  + ' --xe ' + xe  + ' --yb ' + yb  + ' --ye ' + ye  +
#             ' --zb ' + zb  +' --ze ' + ze  +' --save_npy ' + save_npy + ' --save_nii ' + save_nii)
#     f.write("\n date")
#     f.close()


for i in range(len(ori_list) // 5 + 1):

    f = open(slurm_root + '/set' + str(i) + '.sh' , 'w')

    f.write('#!/bin/bash\n')
    f.write('#SBATCH --job-name=set' + str(i) + '\n')
    f.write('#SBATCH --output=' + log_root + '/set' + str(i) + '\n')
    f.write('#SBATCH --ntasks=1\n')
    f.write('#SBATCH --time=07:00:00\n')
    f.write('#SBATCH --mem-per-cpu=11G\n')

    #if i > 1: break
    for j in range(5):
        if 5 * i + j >= len(ori_list): break
        save_npy = save_npy_root + '/' + ori_list[5 * i + j].replace('.nii.gz', '_clean.npy')
        save_nii = save_nii_root + '/' + ori_list[5 * i + j]

        f.write(
            '/home/local/VANDERBILT/gaor2/anaconda3/envs/python36/bin/python /home/local/VANDERBILT/gaor2/code/RNNLung/func/reg_tools/convert_high2low.py '
            + '--ori_path ' + ori_root + '/' + ori_list[5 * i + j] + ' --mask_path ' + mask_root + '/'
            + ori_list[5 * i + j] + ' --xb ' + xb + ' --xe ' + xe + ' --yb ' + yb + ' --ye ' + ye +
            ' --zb ' + zb + ' --ze ' + ze + ' --save_npy ' + save_npy + ' --save_nii ' + save_nii)
        f.write("\n\n")
    f.write("\n date")
    f.close()
