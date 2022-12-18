from glob import glob
import os
import re
import pydicom
import pandas as pd


## use this command to upload: Xnatupload -c /nfs/masi/MCL/xnat/xnat20200110/need_pushagain.csv --deleteAll

data_root = '/nfs/MCL_Jan_21/actual_push'
data_list = os.listdir(data_root)
subject_label, session_label, as_label, as_type, as_description, fpath = [], [], [], [], [], []

for i in range(len(data_list)):
    tmp_list = re.split('[_-]', data_list[i])
    subject_label.append(tmp_list[0])
    session_label.append(data_list[i])

    item_list = os.listdir(data_root + '/' + data_list[i] + '/SCANS')
    print(data_list[i])
    assert len(item_list) == 1

    deep_item = data_root + '/' + data_list[i] + '/SCANS/' + item_list[0] + '/DICOM'
    dcm_list = glob(os.path.join(deep_item, '*.dcm'))
    a = pydicom.dcmread(dcm_list[0])
    desc = a.SeriesDescription
    as_label.append(item_list[0])
    as_type.append(desc)
    as_description.append(desc)
    fpath.append(deep_item)

object_type = ['scan'] * len(data_list)
project_id = ['MCL'] * len(data_list)
session_type = ['CT'] * len(data_list)
quality = ['questionable'] * len(data_list)
resource = ['DICOM'] * len(data_list)

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

data.to_csv('/nfs/masi/MCL/xnat/xnat20200121/upload.csv', index=False)