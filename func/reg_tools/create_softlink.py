import os
import re

need_root = '/nfs/masi/SPORE/nifti/combine'

exist_root = ['/nfs/masi/SPORE/registration/affine_niftyreg/SPORE_reg_20200124/interp/ori']

link_root = '/nfs/masi/SPORE/registration/affine_niftyreg/SPORE_reg_20200126/softlink'

need_list = []

need_subj = os.listdir(need_root)

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

for i in range(len(need_subj)):

    sess_list = os.listdir(need_root + '/' + need_subj[i])
    new_sess = [need_subj[i] + 'time' + sess + '.nii.gz' for sess in sess_list]
    need_list += new_sess

print (len(need_list))

exist_list = []

for i in range(len(exist_root)):
    tmp_exist_list = os.listdir(exist_root[i])
    exist_list += tmp_exist_list

print (len(exist_list))

null_list = list(set(need_list) - set(exist_list))

print (len(null_list))

for i in range(len(null_list)):
    tmp_list = re.split('[time]', null_list[i].replace('.nii.gz', ''))
    subj, sess = tmp_list[0], tmp_list[-1]
    makedir(link_root + '/' + subj + '/' + sess)
    #print ("ln -s " + need_root + '/'  + subj + '/' + sess + '/' + null_list[i] + ' ' + link_root + '/' + subj + '/' + sess + '/' + null_list[i])
    #if os.path.exists(link_root + '/' + subj + '/' + sess + '/' + null_list[i]):
    # try:
    #     print ('rm ' + link_root + '/' + subj + '/' + sess + '/' + null_list[i])
    #     os.system('rm ' + link_root + '/' + subj + '/' + sess + '/' + null_list[i])
    # except:
    #     continue
    os.system("ln -s " + need_root + '/'  + subj + '/' + sess + '/' + null_list[i] + ' ' + link_root + '/' + subj + '/' + sess + '/' + null_list[i])