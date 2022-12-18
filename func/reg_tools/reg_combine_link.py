import os
from glob import glob
import re

link_root = '/nfs/masi/MCL/wholebody_nifti/spread_link'

data_root = '/nfs/masi/MCL/wholebody_nifti/combine'

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

## layer to spread

subj_list = os.listdir(data_root)

for i in range(len(subj_list)):
    sess_list = os.listdir(data_root + '/' + subj_list[i])
    for j in range(len(sess_list)):
        item_path = glob(data_root + '/' + subj_list[i] + '/' + sess_list[j] + '/*.nii.gz')
        assert len(item_path) == 1
        item_path = item_path[0]
        item = re.split('/', item_path)[-1]
        makedir(link_root + '/' + item.replace('.nii.gz', ''))
        os.system('ln -s ' + item_path + ' ' + link_root + '/' + item.replace('.nii.gz', '') + '/' + item)




# spread to spread

# data_list = os.listdir(data_root)
#
# for i in range(len(data_list)):
#     os.system("ln -s " + data_root + '/' + data_list[i] + ' ' + link_root + '/' + data_list[i])

