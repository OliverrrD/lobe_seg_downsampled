from glob import glob
import os
import re
import pydicom
import pandas as pd
import pdb


## use this command to upload: Xnatupload -c /nfs/masi/MCL/xnat/xnat20200110/need_pushagain.csv --deleteAll

# before upload, check the dcms in one folder has the same session name.
# moffti doesn't need the --delete


subject_label, session_label, as_label, as_type, as_description, fpath = [], [], [], [], [], []

data_root = '/nfs/masi/MCL/xnat/MOFFTI'
subj_list = os.listdir(data_root)
for subj in subj_list:
    dcm_list = glob(os.path.join(data_root, subj, '*.dcm'))
    assert len(dcm_list) > 0
    print (subj, len(dcm_list))
    for i in range(len(dcm_list)):
        a = pydicom.dcmread(dcm_list[i])
        tmp_name = re.split('[ ,:;]',a[0x10,0x4000].value.strip())[-1]
        tmp_name_vec = re.split('[-_]', tmp_name)
        session_label.append(tmp_name_vec[0] + '_' + tmp_name_vec[1])
        subj_name = tmp_name_vec[0]
        subject_label.append(subj_name)
        #session_label.append(tmp_name)
        try:
            desc = a[0x08, 0x103e].value
            as_type.append(desc)
            as_description.append(desc)
        except:
            #pdb.set_trace()
            #as_label.append('missed')
            as_type.append('missed')
            as_description.append('missed')
            print (data_list[i], sess)
        try:
            as_label.append(a[0x20,0x11].value)
        except:
            as_label.append('as_label_miss')

        fpath.append( dcm_list[i])
    #break

# for i in range(len(data_list)):
    
    
# #     session_label.append(data_list[i])

#     sess_list = os.listdir(data_root + '/' + data_list[i])
#     for sess in sess_list:
        
#         item_list = os.listdir(data_root + '/' + data_list[i] + '/' + sess)
#         for item in item_list:
            
            
#             deep_item = data_root + '/' + data_list[i] + '/' + sess + '/' + item
#             dcm_list = glob(os.path.join(deep_item, '*.dcm'))
#             if len(dcm_list) > 0:
#                 subject_label.append(data_list[i])
#                 a = pydicom.dcmread(dcm_list[0])
#                 tmp_name = re.split('[ ,:;]',a[0x10,0x4000].value.strip())[-1]
#                 session_label.append(tmp_name)
#                 try:
#                     desc = a[0x08, 0x103e].value
#                     as_type.append(desc)
#                     as_description.append(desc)
#                 except:
#                     #pdb.set_trace()
#                     #as_label.append('missed')
#                     as_type.append('missed')
#                     as_description.append('missed')
#                     print (data_list[i], sess)
#                 try:
#                     as_label.append(a[0x20,0x11].value)
#                 except:
#                     as_label.append('as_label_miss')

#                 fpath.append(deep_item)

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

data.to_csv('/nfs/masi/MCL/xnat/xnat20200816_moffti/upload.csv', index=False)