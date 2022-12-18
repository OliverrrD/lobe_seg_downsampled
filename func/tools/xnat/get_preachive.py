import os
'''
This file is to submit data to xnat with subject_id and session_id
'''
dicom_root = '/share5/gaor2/data/MCL/Xnat0926/MCL/31965730790_31965730790-20100316/31965730790_31965730790-20100316/2-x-INSPIRATION-x-INSPIRATION/DICOM'
    #'/share5/gaor2/data/MCL/prearchive_test/31965730790_31965730790-20100316/2-x-INSPIRATION-x-INSPIRATION/DICOM'
Subject_id = 'test'
Session_id = 'test'

dicom_list = os.listdir(dicom_root)
for i in range(len(dicom_list)):
    dicom_path = dicom_root + '/' + dicom_list[i]
    print (dicom_path)
    os.system('dcmodify -nb -i "(0010,4000)=Project:MCL Subject:'+Subject_id+' Session:'+Session_id+'" '+ dicom_path)
    os.system('storescu -aec VandyXNAT xnat.vanderbilt.edu 8104 ' + dicom_path)