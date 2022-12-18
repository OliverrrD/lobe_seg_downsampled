import os
import re
def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

slurm_root = '/nfs/masi/MCL/wholebody_nifti/slurm/script'

save_root = '/nfs/masi/MCL/wholebody_nifti/slurm/log'
crop_root = '/nfs/masi/MCL/wholebody_nifti/crop'

sl_list = os.listdir(slurm_root)

for i in range(len(sl_list)):
    if i < 201: continue
    #if i > 200: break
    print (i, len(sl_list))
    tmp_vec =re.split('[time]' , sl_list[i].replace('.sh', ''))
    subj, sess = tmp_vec[0], tmp_vec[-1]
    print (crop_root + '/' + subj + '/' + sess)
    makedir(crop_root + '/' + subj + '/' + sess)

    # if i % 3 == 0:
    #     print ('sh ' +  slurm_root + '/' + sl_list[i] + ' >> ' +slurm_root + '/' + sl_list[i].replace('.sh', '.txt') + ' 2>&1 &')
    os.system('sh ' +  slurm_root + '/' + sl_list[i] )