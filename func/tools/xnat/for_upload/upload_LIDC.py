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

data_root = '/nfs/masi/LungData_public/LIDC/LIDC'
subj_list = os.listdir(data_root)
for subj in subj_list:
    sess_list = os.listdir(data_root + '/' + subj)
    
    for i in range(len(sess_list)):
        print (subj, sess_list[i].replace(' ', '-'))
        new_name = sess_list[i].replace(' ', '-')
        os.rename(data_root + '/' + subj + '/' + sess_list[i], data_root + '/' + subj + '/' + new_name)
        src_list = os.listdir(data_root + '/' + subj + '/' + new_name)
        for src in src_list:
            new_src = src.replace(' ', '_').replace('-', '_')
            os.rename(data_root + '/' + subj + '/' + new_name + '/' + src, data_root + '/' + subj + '/' + new_name + '/' + new_src)
            dcmlist = glob(os.path.join(data_root + '/' + subj + '/' + new_name + '/' + new_src, '*.dcm'))
            for dcm_path in dcmlist:
                session_label.append(subj + '_' + new_name)
                subject_label.append(subj)
        #session_label.append(tmp_name)
                as_type.append('missed')
                as_description.append('missed')
                as_label.append(new_src)
                fpath.append(dcm_path)

object_type = ['scan'] * len(subject_label)
project_id = ['LIDC_IDRI'] * len(subject_label)
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

data.to_csv('/nfs/masi/LungData_public/LIDC/upload_to_XNAT2.csv', index=False)