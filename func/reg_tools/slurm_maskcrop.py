import os

ori_root = '/nfs/masi/SPORE/registration/affine_niftyreg/SPORE_reg_20200126/interp/ori'
mask_root = '/nfs/masi/SPORE/registration/affine_niftyreg/SPORE_reg_20200126/interp/mask'

slurm_root = '/nfs/masi/SPORE/registration/affine_niftyreg/SPORE_reg_20200126/After_REG/slurm/script'
log_root = '/nfs/masi/SPORE/registration/affine_niftyreg/SPORE_reg_20200126/After_REG/slurm/log'

save_nii_root = '/nfs/masi/SPORE/registration/affine_niftyreg/SPORE_reg_20200126/After_REG/reg_crop'
save_npy_root = '/nfs/masi/SPORE/registration/affine_niftyreg/SPORE_reg_20200126/After_REG/REG_DSB/prep'

xb, xe, yb, ye, zb, ze = '80', '460', '145', '399', '110', '435'


    #80, 460, 145, 399, 110, 435,     # 20200115 for 1x1x1


    #'83', '457', '148', '391', '68', '412' # this is getting from the fixed image
    # '80', '460', '145', '399', '110', '435' # this is the actually use for 1x1x1 image, new pipeline

ori_list = os.listdir(ori_root)


for i in range(len(ori_list)):

    #if i > 1: break
    save_npy = save_npy_root + '/' + ori_list[i].replace('.nii.gz', '_clean.npy')
    save_nii = save_nii_root + '/' + ori_list[i]

    f = open(slurm_root + '/' + ori_list[i].replace('.nii.gz', '.sh'), 'w')


    f.write('#!/bin/bash\n')
    f.write('#SBATCH --job-name=' + ori_list[i]  + '\n')
    f.write('#SBATCH --output=' + log_root + '/' + ori_list[i].replace('.nii.gz', '.txt') + '\n')
    f.write('#SBATCH --ntasks=1\n')
    f.write('#SBATCH --time=07:00:00\n')
    f.write('#SBATCH --mem-per-cpu=11G\n')


    f.write('/home/local/VANDERBILT/gaor2/anaconda3/envs/python36/bin/python /home/local/VANDERBILT/gaor2/code/RNNLung/func/reg_tools/get_lung_afterreg.py '
            + '--ori_path ' + ori_root + '/' + ori_list[i] + ' --mask_path ' + mask_root + '/'
            + ori_list[i] + ' --xb ' + xb  + ' --xe ' + xe  + ' --yb ' + yb  + ' --ye ' + ye  +
            ' --zb ' + zb  +' --ze ' + ze  +' --save_npy ' + save_npy + ' --save_nii ' + save_nii)
    f.write("\n date")
    f.close()
