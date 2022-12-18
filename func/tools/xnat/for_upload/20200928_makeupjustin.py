from glob import glob
import os
import re
import pydicom
import pandas as pd
import pdb


## use this command to upload: Xnatupload -c /nfs/masi/MCL/xnat/xnat20200110/need_pushagain.csv --deleteAll or --delete

# before upload, check the dcms in one folder has the same session name.
# moffti doesn't need the --delete


subject_label, session_label, as_label, as_type, as_description, fpath = [], [], [], [], [], []

f = open('/nfs/masi/MCL/xnat/xnat20200928_makeupjustin/need_sess_path_step1.txt')
lines = f.readlines()
lines = [line.strip() for line in lines]


for idx in range(len(lines)):
    sess = lines[idx]
    if idx % 10 != 9:
        continue
    item_list = os.listdir(sess)
    sess_name = sess.split('/')[-1]
    subj_name = sess.split('/')[-2]
    for item in item_list:
        dcm_list = glob(os.path.join(sess,item, 'DICOM', '*.dcm'))
        assert len(dcm_list) > 0
        
        for i in range(len(dcm_list)):
            a = pydicom.dcmread(dcm_list[i])
            tmp_name = re.split('[ ,:;]',a[0x10,0x4000].value.strip())[-1]
            tmp_name_vec = re.split('[-_]', tmp_name)
            session_label.append(sess_name)
            
            subject_label.append(subj_name)
            
            try:
                desc = a[0x08, 0x103e].value
                as_type.append(desc)
                as_description.append(desc)
            except:
                #pdb.set_trace()
                #as_label.append('missed')
                as_type.append('missed')
                as_description.append('missed')
                print (sess)
            try:
                as_label.append(a[0x20,0x11].value)
            except:
                as_label.append('as_label_miss')

            fpath.append( dcm_list[i])
        print (sess, item, len(dcm_list))
#         break
#     break

        #print (sess, item, len(dcm_list))


object_type = ['scan'] * len(subject_label)
project_id = ['MCL'] * len(subject_label)
session_type = ['CT'] * len(subject_label)
quality = ['questionable'] * len(subject_label)
resource = ['DICOM'] * len(subject_label)

data = pd.DataFrame()

data['object_type'] = object_type
data['project_id'] = project_id
data['subject_label'] = subject_label
data['session_type'] = session_type
data['session_label'] = session_label
data['as_label'] = as_label
data['as_type'] = as_type
data['as_description'] = as_description
data['quality'] = quality
data['resource'] = resource
data['fpath'] = fpath

data.to_csv('/nfs/masi/MCL/xnat/xnat20200928_makeupjustin/upload_step1_9.csv', index=False)