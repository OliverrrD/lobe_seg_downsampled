import os
import pandas as pd

in_root = '/nfs/masi/gaor2/data/Reg_formedia/resample/proj'

out_root = '/nfs/masi/gaor2/data/Reg_formedia/reg_out/proj'

slurm_root = '/nfs/masi/gaor2/data/Reg_formedia/script/reg/slurm/proj'

log_root = '/nfs/masi/gaor2/data/Reg_formedia/script/reg/log/proj'

csv_path = '/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/data/ThreeSet/registration/proj_mov_ref.csv'

df = pd.read_csv(csv_path)

for i, item in df.iterrows():

    f = open(slurm_root + '/' + item['mov'] +'.sh', 'w')

    f.write('#!/bin/bash\n')
    f.write('#SBATCH --job-name=' + item['mov']  + '\n')
    f.write('#SBATCH --output=' + log_root + '/' + item['mov'] + '.txt' + '\n')
    f.write('#SBATCH --ntasks=1\n')
    f.write('#SBATCH --cpus-per-task=4\n')
    f.write('#SBATCH --time=07:00:00\n')
    f.write('#SBATCH --exclude=melvins,pelican,felakuti,masi-21,slayer\n')
    f.write('#SBATCH --mem-per-cpu=11G\n')
    ref_item = in_root + '/' + item['ref'] + '.nii.gz'
    mov_item = in_root + '/' + item['mov'] + '.nii.gz'
    out_item = out_root + '/' + item['mov'] + '.nii.gz'
    f.write('/home/local/VANDERBILT/gaor2/bin_tool/reg_aladin -ln 3 -ref ' + ref_item + ' -flo '+ mov_item + ' -res ' + out_item)
    
    f.write("\n date")
    f.close()

    
# ori_list = os.listdir(ori_root)

# for i in range(len(ori_list)):

#     f = open(slurm_root + '/' + ori_list[i].replace('.nii.gz', '.sh'), 'w')

#     f.write('#!/bin/bash\n')
#     f.write('#SBATCH --job-name=' + ori_list[i]  + '\n')
#     f.write('#SBATCH --output=' + log_root + '/' + ori_list[i].replace('.nii.gz', '.txt') + '\n')
#     f.write('#SBATCH --ntasks=1\n')
#     f.write('#SBATCH --time=07:00:00\n')
#     f.write('#SBATCH --mem-per-cpu=11G\n')

#     f.write('/home/local/VANDERBILT/xuk9/src/c3d/bin/c3d ' + ori_root + '/' + ori_list[i] + ' -resample-mm 1x1x1mm -o '+ new_root + '/' + ori_list[i])
    
#     f.write("\n date")
#     f.close()