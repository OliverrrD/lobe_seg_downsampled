# for folder all subjects
import pandas as pd
import pydicom
import os
from glob import glob
import re

data_root = '/nfs/masi/MCL/xnat/DECAMP'
data_list = os.listdir(data_root)
for subj in data_list:
    
    subj_path = data_root + '/' + subj
    sess_list = os.listdir(subj_path)
    for sess in sess_list:
        item_list = os.listdir(subj_path + '/' + sess)
        sess_name = None
        flag = 1
        for item in item_list:
            deep_item = subj_path + '/' + sess + '/' + item
            dcm_list = glob(os.path.join(deep_item, '*.dcm'))

            for dcm in dcm_list:
                a = pydicom.dcmread(dcm)
                tmp_name = re.split('[ ,:;]',a[0x10,0x4000].value.strip())[-1]
                if sess_name == None:
                    sess_name = tmp_name
                else:
                    try:
                        assert sess_name == tmp_name
                    except:
                        print (sess_name, tmp_name)
                        print (sess)
                        print (item)
                        print (dcm)
                        flag = 0
                        break
        if flag == 1 and sess_name != None:
            print (subj_path + '/' + sess, subj_path + '/' + sess_name)
            os.rename(subj_path + '/' + sess, subj_path + '/' + sess_name)