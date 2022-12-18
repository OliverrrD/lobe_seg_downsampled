import os

ori_root = '/nfs/masi/MCL/wholebody_nifti/combine'

new_root = '/nfs/masi/MCL/wholebody_nifti/mask'

crop_root = '/nfs/masi/MCL/wholebody_nifti/crop'

slurm_root = '/nfs/masi/MCL/wholebody_nifti/slurm/script'

log_root = '/nfs/masi/MCL/wholebody_nifti/slurm/log'

subj_list = os.listdir(ori_root)

for i in range(len(subj_list)):

    subj_path = ori_root + '/' + subj_list[i]
    sess_list = os.listdir(subj_path)

    for j in range(len(sess_list)):

        f = open(slurm_root + '/' + subj_list[i] + 'time' + sess_list[j] + '.sh', 'w')

        f.write('#!/bin/bash\n')
        f.write('#SBATCH --job-name=' + subj_list[i] + 'time' + sess_list[j]+ '\n')
        f.write('#SBATCH --output=' + log_root + '/' + subj_list[i] + 'time' + sess_list[j] + '.txt' + '\n')
        f.write('#SBATCH --ntasks=1\n')
        f.write('#SBATCH --time=07:00:00\n')
        f.write('#SBATCH --mem-per-cpu=11G\n')
       # f.write("export FREESURFER_HOME=/nfs/masi/xuk9/local/freesurfer\n")
       # f.write("source $FREESURFER_HOME/SetUpFreeSurfer.sh\n")

        ori_path = ori_root + '/' + subj_list[i] + '/' + sess_list[j] + '/' + subj_list[i] + 'time' + sess_list[j] + '.nii.gz'
        mask_path = new_root + '/' + subj_list[i] + '/' + sess_list[j] + '/' + subj_list[i] + 'time' + sess_list[j] + '.nii.gz'
        crop_path = crop_root + '/' + subj_list[i] + '/' + sess_list[j] + '/' + subj_list[i] + 'time' + sess_list[j] + '.nii.gz'

        f.write(
            '/home/local/VANDERBILT/gaor2/anaconda3/envs/python37/bin/python /home/local/VANDERBILT/gaor2/code/RNNLung/func/tools/seg_lung.py --ori '
            + ori_path + ' --out ' + mask_path)
        f.write('\n')
        f.write('/home/local/VANDERBILT/gaor2/anaconda3/envs/python37/bin/python /home/local/VANDERBILT/gaor2/code/RNNLung/func/tools/seg_roi.py --mask '
            + mask_path + ' --roi ' + crop_path + ' --img ' + ori_path)

        f.write("\n date")
        f.close()
        #item = os.listdir(subj_path + '/' + sess_list)

