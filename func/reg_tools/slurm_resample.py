import os

ori_root = '/nfs/masi/MCL/registration/affine_niftyreg_0/interp/ori'

new_root = '/nfs/masi/MCL/registration/affine_niftyreg_0/interp/ori_1x1x1'

slurm_root = '/nfs/masi/MCL/registration/affine_niftyreg_0/interp/resample_slurm/script'

log_root = '/nfs/masi/MCL/registration/affine_niftyreg_0/interp/resample_slurm/log'

ori_list = os.listdir(ori_root)

for i in range(len(ori_list)):

    f = open(slurm_root + '/' + ori_list[i].replace('.nii.gz', '.sh'), 'w')

    f.write('#!/bin/bash\n')
    f.write('#SBATCH --job-name=' + ori_list[i]  + '\n')
    f.write('#SBATCH --output=' + log_root + '/' + ori_list[i].replace('.nii.gz', '.txt') + '\n')
    f.write('#SBATCH --ntasks=1\n')
    f.write('#SBATCH --time=07:00:00\n')
    f.write('#SBATCH --mem-per-cpu=11G\n')
    f.write("export FREESURFER_HOME=/nfs/masi/xuk9/local/freesurfer\n")
    f.write("source $FREESURFER_HOME/SetUpFreeSurfer.sh\n")

    f.write('/nfs/masi/xuk9/local/freesurfer/bin/mri_convert ' + ori_root + '/' + ori_list[i] + ' '+ new_root + '/' + ori_list[i] + ' -vs 1.0 1.0 1.0')
    f.write("\n date")
    f.close()
