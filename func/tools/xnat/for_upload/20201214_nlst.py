from glob import glob
import os
import re
import pydicom
import pandas as pd
import pdb

subject_label, session_label, as_label, as_type, as_description, fpath = [], [], [], [], [], []
df = pd.read_excel('/nfs/masi/MCL/file/NLST_related/Khushbu_NLST_dataset_for_LungSpore.xlsx')
ksbu_ID = df['PATIENT    ID'].tolist()
ksbu_ID = [str(i) for i in ksbu_ID]

data_root = '/nfs/masi/NLST/ORI_DICOM/NLST_cancer'
subj_list = os.listdir(data_root)
cnt = 0 
for subj in ksbu_ID:
    cnt += 1
    if cnt < 100:
        continue
    if cnt >= 130: 
        break
    if subj not in subj_list:
        print (subj + ' is not found!')
        continue
        
    sess_list = os.listdir(data_root + '/' + subj)
    
    for sess in sess_list:
        scan_list = os.listdir(data_root + '/' + subj + '/' + sess)
        for scan in scan_list:
            #pdb.set_trace()
            dcm_list = glob(os.path.join(data_root, subj, sess, scan, '*.dcm'))
    
            print (cnt, subj, len(dcm_list))
            for i in range(len(dcm_list)):
                a = pydicom.dcmread(dcm_list[i])
                #tmp_name = re.split('[ ,:;]',a[0x10,0x4000].value.strip())[-1]
                #tmp_name_vec = re.split('[-_]', tmp_name)
                session_label.append(a[0x10, 0x20].value + '_' + a[0x08, 0x20].value)
                
                subject_label.append(a[0x10, 0x20].value)
                #session_label.append(tmp_name)
                try:
                    desc = 'NA'
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

data.to_csv('/nfs/masi/MCL/xnat/xnat20201214_nlst/upload_100toall.csv', index=False)