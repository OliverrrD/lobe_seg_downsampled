import os

slurm_root = '/nfs/masi/gaor2/data/Reg_formedia/script/reg/slurm/proj'
slurm_list = os.listdir(slurm_root)
save_root = '/nfs/masi/gaor2/data/Reg_formedia/reg_out/proj'

for i in range(len(slurm_list)):
    # if i < 1000:
    #     continue
    if os.path.exists(save_root + '/' + slurm_list[i].replace('.sh', '.nii.gz')):
        continue
    print ("sbatch " + slurm_root + '/' + slurm_list[i] + '\n')
    os.system("sbatch " + slurm_root + '/' + slurm_list[i] + '\n')



# slurm_root = '/nfs/masi/gaor2/data/Reg_formedia/script/resample/slurm/proj'
# log_root = 
# slurm_list = os.listdir(slurm_root)

# for i in range(len(slurm_list)):
#     if i / 600 == 0: 
#         ID = slurm_list[i].replace('.sh', '')
        
#         print ("sh " + slurm_root + "/" + ID + ".sh > " + log_root + "/" + ID + ".txt 2>&1 &")
#         os.system("sh " + slurm_root + "/" + ID + ".sh > " + log_root + "/" + ID + ".txt 2>&1 &")


