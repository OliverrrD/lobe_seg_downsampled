import os
import pandas as pd
#import nibabel as nib
import numpy as np
from skimage import transform, util
import re
import matplotlib.pyplot as plt
import SimpleITK as sitk
#import nibabel as nib
import shutil
import matplotlib.patches as patches
from glob import glob
import h5py
#from clinical_tool import *

'''
This file includes the functions of general operations to data. Such as change the change the format of data, change the folder structure. 
'''

def dcm2nii(src_root, dst_root):
    '''
    This function transfer the dcm form data to nii form
    :param src_root: The original data 's folder (root)
    :param dst_root: The target data 's folder
    :return:
    '''
    cmd_pre = "dcm2niix -m y "                                     #    "dcm2nii -r n "           # dcm2niix -m y                            
    cmd = cmd_pre + "-o " + dst_root + " " + src_root
    os.system(cmd)
    print (cmd)

def dcm2nii_second(err_txt):
    '''
    dcm2nii_second('/share5/gaor2/data/txt/new_err2.txt')
    because there are some unexpected char in the path, we have to remove these unexpected char and do the dcm2nii again.
    :param err_txt:
    :return:
    '''
    f = open(err_txt, 'r')
    lines = f.readlines()
    for i in range(len(lines)):
        print (i)
        line_vec = lines[i].split('----------')
        pre_folder = re.sub('[()]', '', line_vec[0])
        os.rename(line_vec[0], pre_folder)
        src_root = pre_folder + '/DICOM'
        dst_root = pre_folder + '/new_NIFTI'

        dcm2nii(src_root, dst_root)

def get_trdata_second(ori_err_txt, new_root, copy_txt):
    '''
    get_trdata_second('/share5/gaor2/data/txt/new_err2.txt', '/share5/gaor2/data/MCL_data2', '/share5/gaor2/data/txt/new_copy2_second.txt')
    :param ori_err_txt:
    :param new_root:
    :param copy_txt:
    :return:
    '''
    f = open(ori_err_txt, 'r')
    f_copy = open(copy_txt, 'w')
    lines = f.readlines()
    for i in range(len(lines)):
        print(i)
        line_vec = lines[i].split('----------')
        pre_folder = re.sub('[()]', '', line_vec[0])
        info = pre_folder.split('/')[-2]
        info_vec = re.split('[-_]', info)
        subj_id = info_vec[0]
        sess_id = info_vec[-1]
        if not os.path.exists(new_root + '/' + subj_id):
            os.mkdir(new_root + '/' + subj_id)
        if not os.path.exists(new_root + '/' + subj_id + '/' + sess_id):
            os.mkdir(new_root + '/' + subj_id + '/' + sess_id)
        os.system(
            'cp ' + pre_folder + '/new_NIFTI/*' + ' ' + new_root + '/' + subj_id + '/' + sess_id + '/' + subj_id + 'time' + sess_id + 'item00'  + '.nii.gz')
        print ('cp ' + pre_folder + '/new_NIFTI/*' + ' ' + new_root + '/' + subj_id + '/' + sess_id + '/' + subj_id + 'time' + sess_id + 'item00'  + '.nii.gz')
        f_copy.write(pre_folder + '/new_NIFTI/*' + '==========' + new_root + '/' + subj_id + '/' + sess_id + '/' + subj_id + 'time' + sess_id + 'item00'  + '.nii.gz')
    f_copy.close()


def dcm2nii_folder(src_root, dst_root):
    cmd_pre = "dcm2nii -r n "
    src_dir = os.listdir(src_root)
    print(len(src_dir))
    for tmp_src in src_dir:
        tmp_dst = dst_root + '/' + tmp_src
        if not os.path.exists(tmp_dst):
            os.mkdir(tmp_dst)
        #cmd = cmd_pre + "-o " + tmp_dst + " " + src_root + '/' + tmp_src
        dcm2nii(src_root + '/' + tmp_src, tmp_dst)
        # os.system(cmd)
        # print(cmd)

def dcm2nii_MCL0(MCL_root):
    '''
    exp: dcm2nii_MCL('/share5/gaor2/data/MCL_subjects/MCL')
    :param MCL_root:
    :return:
    '''
    subj_list = os.listdir(MCL_root)
    for subj in subj_list:
        print (subj)
        subj_path = MCL_root + '/' + subj
        sess_list = os.listdir(subj_path)
        for sess in sess_list:
            sess_path = subj_path + '/' + sess
            fder_list = os.listdir(sess_path)
            for fder in fder_list:
                fder_path = sess_path +'/' + fder
                item_list = os.listdir(fder_path)
                #if 'NIFTI' not in item_list:
                if 'DICOM' in item_list:
                        
                        mkdir(fder_path + '/new_NIFTI')
                        dcm2nii(fder_path + '/DICOM', fder_path + '/new_NIFTI')

def dcm2nii_SPORE0(SPORE_root):
    subj_list = os.listdir(SPORE_root)
    for subj in subj_list:
        print (subj)
        #if (subj == '40593716009'): continue
        subj_path = SPORE_root + '/' + subj
        sess_list = os.listdir(subj_path)
        for sess in sess_list:
            sess_path = subj_path + '/' + sess
            fder_list = os.listdir(sess_path)
            for i in range(len(fder_list)):
                fder = fder_list[i]
                ori_fder_path = sess_path +'/' + fder
                fder_path = sess_path + '/file' + str(i)
                #print ('mv ' + ori_fder_path + ' ' + fder_path)
                os.rename(ori_fder_path ,fder_path)
                print (fder_path)
                item_list = os.listdir(fder_path)
                #if 'NIFTI' not in item_list:
                if 'DICOM' in item_list:
                        mkdir(fder_path + '/new_NIFTI')
                        if len(os.listdir(fder_path + '/new_NIFTI')) != 0:
                            continue

                        dcm2nii(fder_path + '/DICOM', fder_path + '/new_NIFTI')

def get_trdata_SPORE(data_root, new_root):
    subj_list = os.listdir(data_root)
    for i in range(len(subj_list)):
        print (i, subj_list[i])
        subj_path = os.path.join(data_root, subj_list[i])
        sess_list = os.listdir(subj_path)
        subj_id = subj_list[i].split('_')[1]
        mkdir(new_root + '/' + subj_id)
        for j in range(len(sess_list)):
            sess_path = os.path.join(subj_path, sess_list[j])
            if len(os.listdir(sess_path)) != 1:
                print (sess_path, 'len != 1')
                continue
            nifti_path = sess_path + '/' + os.listdir(sess_path)[0] + '/new_NIFTI'
            sess_id = re.split('[_-]', sess_list[j])[2]
            mkdir(new_root + '/' + subj_id + '/' + sess_id)
            if (len(os.listdir(new_root + '/' + subj_id + '/' + sess_id)) > 0):
                print (new_root + '/' + subj_id + '/' + sess_id, ' already completed')
                continue
            data_list = glob(os.path.join(nifti_path, '*.nii.gz'))
            size_list = []
            for k in range(len(data_list)):
                item = data_list[k]
                size_list.append(os.path.getsize(item))
            max_index = size_list.index(max(size_list))
            os.rename(data_list[max_index], nifti_path + '/max.nii.gz') # need rename becasue some name contain '(,' cannot be copied

            new_name = subj_id + 'time' + sess_id + '.nii.gz'
            print ('mv ' + nifti_path + '/max.nii.gz' + ' ' + new_root + '/' + subj_id + '/' + sess_id + '/' + new_name)
            os.system('mv ' + nifti_path + '/max.nii.gz' + ' ' + new_root + '/' + subj_id + '/' + sess_id + '/' + new_name)
            
def get_trdata_MCL(data_root, new_root):
    subj_list = os.listdir(data_root)
    for i in range(len(subj_list)):
        print (i, subj_list[i])
        subj_path = os.path.join(data_root, subj_list[i])
        sess_list = os.listdir(subj_path)
        #subj_id = subj_list[i].split('_')[1]
        #mkdir(new_root + '/' + subj_id)
        for j in range(len(sess_list)):
            sess_path = os.path.join(subj_path, sess_list[j])
            
            if len(os.listdir(sess_path)) != 1:
                print (sess_path, 'len != 1')
                continue
            nifti_path = sess_path + '/' + os.listdir(sess_path)[0] + '/new_NIFTI'
            subj_id = re.split('[_-]', sess_list[j])[0]
            sess_id = re.split('[_-]', sess_list[j])[-1]
            mkdir(new_root + '/' + subj_id + '/' + sess_id)
            if (len(os.listdir(new_root + '/' + subj_id + '/' + sess_id)) > 0):
                print (new_root + '/' + subj_id + '/' + sess_id, ' already completed')
                continue
            data_list = glob(os.path.join(nifti_path, '*.nii.gz'))
            size_list = []
            for k in range(len(data_list)):
                item = data_list[k]
                size_list.append(os.path.getsize(item))
            max_index = size_list.index(max(size_list))
            os.rename(data_list[max_index], nifti_path + '/max.nii.gz') # need rename becasue some name contain '(,' cannot be copied
            
            new_name = subj_id + 'time' + sess_id + '.nii.gz'
            print ('mv ' + nifti_path + '/max.nii.gz' + ' ' + new_root + '/' + subj_id + '/' + sess_id + '/' + new_name)
            os.system('mv ' + nifti_path + '/max.nii.gz' + ' ' + new_root + '/' + subj_id + '/' + sess_id + '/' + new_name)
                        
def xnat_info(data_root):
    '''
    This function is to see if the data from xnat have convert to nifti successfully
    '''
    subj_list = os.listdir(data_root)
    for i in range(len(subj_list)):
        subj_path = os.path.join(data_root, subj_list[i])
        sess_list = os.listdir(subj_path)
        for j in range(len(sess_list)):
            sess_path = os.path.join(subj_path, sess_list[j])
            if len(os.listdir(sess_path)) != 1:
                print (sess_path, 'len != 1')
                continue
            nifti_path = sess_path + '/' + os.listdir(sess_path)[0] + '/new_NIFTI'
            if len(os.listdir(nifti_path)) == 0:
                print (nifti_path, ' no nifit')
            if len(os.listdir(nifti_path)) > 2:
                print (nifti_path, ' more nifit')
                        
def dcm2nii_NLST(NLST_root, save_root):
    subj_list = os.listdir(NLST_root)
    for i in range(len(subj_list)):
        subj = subj_list[i]
        print (i, subj)                   # 1001
        subj_path = NLST_root + '/' + subj
        sess_list = os.listdir(subj_path)
        for sess in sess_list:
            sess_path = subj_path + '/' + sess
            print (sess_path)
            fder_list = os.listdir(sess_path)
            for fder in fder_list:
                fder_path = sess_path +'/' + fder
                dcm_list = os.listdir(fder_path)
                if (len(dcm_list) < 80): continue
                save_path = save_root + '/' + subj + '/' + sess
                mkdir(save_path)
                if len(glob(os.path.join(save_path, '*.nii.gz'))) != 0:
                    print (save_path, ' already contain nii.gz')
                    continue
                print ('======', 'dcm2nii -r n -o ' + save_path + ' ' + fder_path)
                os.system('dcm2nii -r n -o ' + save_path + ' ' + fder_path)

                
def dcm2nii_condition_NLST(NLST_root, save_root, csv_path):
    '''
    dcm2nii_condition_NLST('/media/gaor2/MASIDATA1/NLST/data/NLST', '/media/gaor2/MASIDATA1/NLST/data/NLST_nifti', '/media/gaor2/MASIDATA1/NLST/nocancer_all.csv')
    '''
    subj_list = os.listdir(NLST_root)
    df = pd.read_csv(csv_path)
    pid_list = df['pid'].tolist()
    pid_list = [str(i) for i in pid_list]
#    print (pid_list)
    for i in range(len(pid_list)):
#         if i > 1:
#             break
        subj = pid_list[i]
        print (subj)
        if subj not in pid_list:
            print (subj + ' is not negative')
            continue
        print (i, subj)                   # 1001
        subj_path = NLST_root + '/' + subj
        sess_list = os.listdir(subj_path)
        for sess in sess_list:
            sess_path = subj_path + '/' + sess
            print (sess_path)
            fder_list = os.listdir(sess_path)
            for fder in fder_list:
                fder_path = sess_path +'/' + fder
                dcm_list = os.listdir(fder_path)
                if (len(dcm_list) < 60): continue
                
                save_path = save_root + '/' + subj + '/' + sess
                if len(glob(os.path.join(save_path, '*.nii.gz'))) != 0:
                    print (save_path, ' already contain nii.gz')
                    continue
                mkdir(save_path)
                print ('======', 'dcm2nii -r n -o ' + save_path + ' ' + fder_path)
                os.system('dcm2nii -r n -o ' + save_path + ' ' + fder_path)
    
def get_trdata(ori_root, new_root, copy_txt, nonifti_txt,  err_txt):
    '''
    exp:get_trdata('/share5/gaor2/data/MCL_subjects/MCL', '/share5/gaor2/data/MCL_data2', '/share5/gaor2/data/txt/new_copy2.txt', '/share5/gaor2/data/txt/new_nonifti2.txt',  '/share5/gaor2/data/txt/new_err2.txt')
    exp: get_trdata('/share5/wangj46/Xnat/MCL', '/share5/gaor2/data/MCL_data', '/share5/gaor2/data/txt/new_copy.txt', '/share5/gaor2/data/txt/new_nonifti.txt',  '/share5/gaor2/data/txt/new_err.txt')

    :param ori_root:
    :param new_root:
    :param copy_txt:
    :param nonifti_txt:
    :param err_txt:
    :return:
    '''
    subj_list = os.listdir(ori_root)
    f_copy = open(copy_txt, 'w')
    f_nonifti = open(nonifti_txt, 'w')
    f_err = open(err_txt, 'w')
    sess_cnt = 0
    for i in range(len(subj_list)):
        subj = subj_list[i]
        subj_id = re.split('[-_]', subj)[0].strip()
#        print ("subject: " + str(i) + '  ' + subj_id)
        subj_path = ori_root + '/' + subj
        sess_list = os.listdir(subj_path)
        for j in range(len(sess_list)):
            sess = sess_list[j]
            sess_id = re.split('[-_]', sess)[-1].strip()
            #print("sess: " + str(j) + '  ' + sess_id)
            sess_path = subj_path + '/' + sess
            fder_list = os.listdir(sess_path)
#             if not os.path.exists(new_root + '/'+ subj_id  + '/' + sess_id):
#                 print (subj_id  + '/' + sess_id)
# #             sess_cnt += 1
# #             print (sess_cnt)
#             continue
            
            for k in range(len(fder_list)):
                fder = fder_list[k]
                fder_path = sess_path + '/' + fder
                try:
                    nifti_list = glob(os.path.join(fder_path + '/new_NIFTI', '*.nii.gz'))    ####### attention !!!!!!!!!!!########
                except:
#                    print (fder_path + '/new_NIFTI ----------is not exist\n')
                    f_nonifti.write(fder_path + '/new_NIFTI ----------is not exist\n')
                    continue

                if len(nifti_list) != 1:
#                    print ('nifti_list', nifti_list)
#                    print('the path: ', fder_path, '---------- has a problem \n')
                    f_err.write(fder_path + '---------- has no or more than one item \n')
                    continue  # I believe here should be continue 09/08
                print (nifti_list[0])
                nifti_path = nifti_list[0]
                if not os.path.exists(new_root + '/'+ subj_id):
                    os.mkdir(new_root + '/'+ subj_id)
                    
#                 if not os.path.exists(new_root + '/'+ subj_id  + '/' + sess_id):
#                     os.mkdir(new_root + '/'+ subj_id  + '/' + sess_id)
#                 else:
#                     #print (new_root + '/'+ subj_id  + '/' + sess_id + ' already exists! ')
#                     continue

                os.system('cp ' + nifti_path + ' ' + new_root + '/'+ subj_id  + '/' + sess_id + '/' + subj_id + 'time' + sess_id + 'item' + str(k).zfill(2) + '.nii.gz')
                print (nifti_path + '==========' + new_root + '/'+ subj_id  + '/' + sess_id + '/' + subj_id + 'time' + sess_id +  'item' + str(k).zfill(2) + '.nii.gz')
                f_copy.write(nifti_path + '==========' + new_root + '/'+ subj_id  + '/' + sess_id + '/' + subj_id + 'time' + sess_id +  'item' + str(k).zfill(2) + '.nii.gz' + '\n')
    f_copy.close()
    f_nonifti.close()
    f_err.close()

def get_trdata1025(ori_root, new_root, copy_txt, nonifti_txt,  err_txt):
    '''
    because the data of xnat1025 name has a different format, so change the code. 
    exp:get_trdata('/share5/gaor2/data/MCL_subjects/MCL', '/share5/gaor2/data/MCL_data2', '/share5/gaor2/data/txt/new_copy2.txt', '/share5/gaor2/data/txt/new_nonifti2.txt',  '/share5/gaor2/data/txt/new_err2.txt')
    exp: get_trdata('/share5/wangj46/Xnat/MCL', '/share5/gaor2/data/MCL_data', '/share5/gaor2/data/txt/new_copy.txt', '/share5/gaor2/data/txt/new_nonifti.txt',  '/share5/gaor2/data/txt/new_err.txt')

    :param ori_root:
    :param new_root:
    :param copy_txt:
    :param nonifti_txt:
    :param err_txt:
    :return:
    '''
    subj_list = os.listdir(ori_root)
    f_copy = open(copy_txt, 'w')
    f_nonifti = open(nonifti_txt, 'w')
    f_err = open(err_txt, 'w')
    sess_cnt = 0
    for i in range(len(subj_list)):
        subj = subj_list[i]
        subj_id = re.split('[-_]', subj)[0].strip()
#        print ("subject: " + str(i) + '  ' + subj_id)
        subj_path = ori_root + '/' + subj
        sess_list = os.listdir(subj_path)
        for j in range(len(sess_list)):
            sess = sess_list[j]
            sess_vec = re.split('[-_]', sess.strip())
            sess_id = ''
            for i in range(1, len(sess_vec)):
                sess_id += sess_vec[i]
            print ('the sess_id is: ', sess_id)
            #sess_id = re.split('[-_]', sess)[-1].strip()
            #print("sess: " + str(j) + '  ' + sess_id)
            sess_path = subj_path + '/' + sess
            fder_list = os.listdir(sess_path)
#             if not os.path.exists(new_root + '/'+ subj_id  + '/' + sess_id):
#                 print (subj_id  + '/' + sess_id)
# #             sess_cnt += 1
# #             print (sess_cnt)
#                 continue
            
            for k in range(len(fder_list)):
                fder = fder_list[k]
                fder_path = sess_path + '/' + fder
                try:
                    nifti_list = os.listdir(fder_path + '/new_NIFTI')    ####### attention !!!!!!!!!!!########
                except:
                    f_nonifti.write(fder_path + '/new_NIFTI ----------is not exist\n')
                    continue

                if len(nifti_list) != 1:
                    print('the path: ', fder_path, '---------- has a problem \n')
                    f_err.write(fder_path + '---------- has no or more than one item \n')
                    continue  # I believe here should be continue 09/08
                nifti_path = fder_path + '/new_NIFTI/' + nifti_list[0]
                if not os.path.exists(new_root + '/'+ subj_id):
                    os.mkdir(new_root + '/'+ subj_id)
                    
                if not os.path.exists(new_root + '/'+ subj_id  + '/' + sess_id):
                    os.mkdir(new_root + '/'+ subj_id  + '/' + sess_id)
                else:
                    #print (new_root + '/'+ subj_id  + '/' + sess_id + ' already exists! ')
                    continue

                os.system('cp ' + nifti_path + ' ' + new_root + '/'+ subj_id  + '/' + sess_id + '/' + subj_id + 'time' + sess_id + 'item' + str(k).zfill(2) + '.nii.gz')
                print (nifti_path + '==========' + new_root + '/'+ subj_id  + '/' + sess_id + '/' + subj_id + 'time' + sess_id +  'item' + str(k).zfill(2) + '.nii.gz')
                f_copy.write(nifti_path + '==========' + new_root + '/'+ subj_id  + '/' + sess_id + '/' + subj_id + 'time' + sess_id +  'item' + str(k).zfill(2) + '.nii.gz' + '\n')
    f_copy.close()
    f_nonifti.close()
    f_err.close()
    
def get_subsess_xnat(data_root, save_path):
    subj_list = os.listdir(data_root)
    sublist = []
    sesslist = []
    sesscnt = []
    for i in range(len(subj_list)):
        subj = subj_list[i]
        subj_id = re.split('[-_]', subj)[0].strip()
        print ("subject: " + str(i) + '  ' + subj_id)
        subj_path = data_root + '/' + subj
        sess_list = os.listdir(subj_path)
        for j in range(len(sess_list)):
            sess = sess_list[j]
            sess_id = re.split('[-_]', sess)[-1].strip()
            print("sess: " + str(j) + '  ' + sess_id)
            sess_path = subj_path + '/' + sess
            fder_list = os.listdir(sess_path)
            sublist.append(subj_id)
            sesslist.append(subj_id + 'time' + sess_id)
            sesscnt.append(len(fder_list))
    sub_set = set(sublist)
    print ('the total subj number is: ', len(sub_set))
    data = pd.DataFrame()
    data['subject'] = sublist
    data['session'] = sesslist
    data['sess_cnt'] = sesscnt
    data.to_csv(save_path)
        
          

def get_id_map(ori_root,  save_txt):
    '''
    get_id_map('/share5/wangj46/Xnat/MCL',  '/share5/gaor2/data/idmap_info.txt')
    :param ori_root:
    :param save_txt:
    :return:
    '''
    dict = {}
    subj_list = os.listdir(ori_root)
    f = open(save_txt, 'w')
    for i in range(len(subj_list)):
        subj = subj_list[i]
        f.write('subj'+ str(i).zfill(5) + ' ' + subj + '\n')
        dict['subj'+ str(i).zfill(5)] = subj
    f.close()
    return dict


def raw2nii(src_root, dst_root):
    '''
    exp: raw2nii('/share5/huoy1/Luna16/seg-lungs-LUNA16', '/share5/huoy1/Luna16/seg-lungs-LUNA16-nii')
    :param src_root:
    :param dst_root:
    :return:
    '''
    print ("convert image from ", src_root, " to ", dst_root)
    img_list = os.listdir(src_root)
    print (len(img_list))
    for i in range(len(img_list)):
        if img_list[i][-4:] != '.mhd':
            continue
        print ('the ', i, ' item')
        itkimage = sitk.ReadImage(src_root + '/' + img_list[i])
        im_array = sitk.GetArrayFromImage(itkimage)
        affine = np.diag([1, 2, 3, 1])
        im_array = nib.Nifti1Image(im_array, affine)
        nib.save(im_array, dst_root + '/' + img_list[i][:-4] + '.nii')


def create_train_data(csv_path, nii_root,  data_name = 'kaggle'):
    '''
    exp: create_train_data('/share5/wangj46/stage1_labels.csv', '/share5/wangj46/kagglenii', '/share5/wangj46/MCLtrain_kaggle')
    :param csv_path:
    :param nii_root:
    :param data_name:
    :return:
    '''
    df = pd.read_csv(csv_path)
    for index, item in df.iterrows():
        if data_name == 'kaggle':
            filepath = nii_root + '/' + item['id']
        if data_name == 'MCL':
            if item['Histologic Type Meta Num'] == 4:
                print ("item['Histologic Type Meta Num'] == 4")
                break
            filepath = nii_root + '/' + item['share3_namepath']

        if (not os.path.exists(filepath)): continue
        filename = filepath + "/" + os.listdir(filepath)[0]
        img = nib.load(filename)
        rawdata = img.get_data()
        rawdata[rawdata <= -1024] = -1024
        rawdata[rawdata >= 1024] = 1024

        new_spacing = [1, 1, 1]
        side = 128
        header = img.header
        spacing = np.array([header['pixdim'][1]] + [header['pixdim'][2]] + [header['pixdim'][3]], dtype=np.float32)
        resize_factor = spacing / new_spacing
        new_shape = np.round(rawdata.shape * resize_factor)

        rawdata = transform.resize(rawdata, new_shape, mode='edge', preserve_range='True')
        maxlen = np.array(max(rawdata.shape)) - np.array(rawdata.shape) + 5

        rawdata = util.pad(rawdata, ((int(np.ceil(maxlen[0] / 2)), int(np.floor(maxlen[0] / 2))),
                                     (int(np.ceil(maxlen[1] / 2)), int(np.floor(maxlen[1] / 2))),
                                     (int(np.ceil(maxlen[2] / 2)), int(np.floor(maxlen[2] / 2)))), 'edge')
        rawdata = transform.resize(rawdata, (side, side, side), mode='edge', preserve_range='True')

        # scale to 0 to 1
        rawdata = (rawdata - rawdata.min()) / (rawdata.max() - rawdata.min())

        rawdata = rawdata.astype('float32')
        rawdata = np.transpose(rawdata, (2, 0, 1))  # to ZXY representation
        #np.save(save_root + '/' + filepath[filepath.rfind("/") + 1:], rawdata)
        return rawdata


def get_id_count(data_root, plot = True):
    '''
    to show how many samples do every id have.
    exp: get_id_count('/share5/wangj46/Xnat/MCL', plot = True)
    :param data_root:
    :param plot:
    :return:
    '''
    id_list = os.listdir(data_root)

    id_num_dict = {}

    for id_string in id_list:
        id = re.split('[-_]', id_string)[0]
        time_list = os.listdir(data_root + '/' + id_string)
        if id not in id_num_dict.keys():
            id_num_dict[id] = 0
        # print (len(time_list))
        id_num_dict[id] += len(time_list)
    #print (id_num_dict)
    if plot == True:
        hist = []
        for key in id_num_dict.keys():
            hist.append(id_num_dict[key])
        plt.hist(hist)
    print ('people with one CT', sum(i == 1 for i in hist ))
    print('people with two CT', sum(i == 2 for i in hist))
    return id_num_dict

def change_data_folder0(ori_root, new_root):
    '''
    exp: change_data_folder('/share5/gaor2/data/MCL/MCL_data2', '/share5/gaor2/data/MCL/MCL_set2')
    '''
    subj_list = os.listdir(ori_root)
    for i in range(len(subj_list)):
        print (subj_list[i])
        if not os.path.exists(new_root + '/' + subj_list[i]):
            os.mkdir(new_root + '/' + subj_list[i])
        subj_path = ori_root + '/' + subj_list[i]
        sess_list = os.listdir(subj_path)
        for j in range(len(sess_list)):
            os.system('cp ' + subj_path + '/' + sess_list[j] + '/* ' + new_root + '/' + subj_list[i])
            


def change_data_folder(ori_root, new_root):
    '''
    Actually this create spread folder
    change_data_folder2('/share5/gaor2/data/MCL/MCL_data2', '/share5/gaor2/data/MCL/MCL_set2_all')
    '''
    subj_list = os.listdir(ori_root)
    for i in range(len(subj_list)):
        print (subj_list[i])
        subj_path = ori_root + '/' + subj_list[i]
        sess_list = os.listdir(subj_path)
        for j in range(len(sess_list)):
            sess_path = subj_path + '/' + sess_list[j]
            item_list = os.listdir(sess_path)
            for k in range(len(item_list)):
                if (item_list[k][-7:] != '.nii.gz'):
                    print (item_list[k])
                    shutil.rmtree(sess_path + '/' + item_list[k])
                    continue
                item_name = item_list[k][:-7]
                if not os.path.exists(new_root + '/' + item_name):
                    os.mkdir(new_root + '/' + item_name)
                #print ('cp ' + sess_path + '/' + item_list[k]+ ' ' + new_root + '/' + item_name + "/")
                os.system('cp ' + sess_path + '/' + item_list[k]+ ' ' + new_root + '/' + item_name)

                
                
def change_NLST_folder(ori_root, new_root):
    subj_list = os.listdir(ori_root)
    for i in range(len(subj_list)):
       
        print (subj_list[i], i)
        subj_path = ori_root + '/' + subj_list[i]
        sess_list = os.listdir(subj_path)
        for j in range(len(sess_list)):
            sess_path = subj_path + '/' + sess_list[j]
            if (len(os.listdir(sess_path)) == 0): 
                print (sess_path, ' has no nifti')
                continue
            item_name = os.listdir(sess_path)[0]
            new_item = subj_list[i] + 'time' + sess_list[j][6:10] # sess_list[j] like 01-02-1999-NLST....
            if not os.path.exists(new_root + '/' + new_item):
                os.mkdir(new_root + '/' + new_item)
                #print ('cp ' + sess_path + '/' + item_list[k]+ ' ' + new_root + '/' + item_name + "/")
            print ('mv ' + sess_path + '/' + item_name + ' ' + new_root + '/' + new_item + '/' + new_item + '.nii.gz')
            os.system('mv ' + sess_path + '/' + item_name + ' ' + new_root + '/' + new_item + '/' + new_item + '.nii.gz')

def nms(output, nms_th):
    if len(output) == 0:
        return output

    output = output[np.argsort(-output[:, 0])]
    bboxes = [output[0]]

    for i in np.arange(1, len(output)):
        bbox = output[i]
        flag = 1
        for j in range(len(bboxes)):
            if iou(bbox[1:5], bboxes[j][1:5]) >= nms_th:
                flag = -1
                break
        if flag == 1:
            bboxes.append(bbox)

    bboxes = np.asarray(bboxes, np.float32)
    return bboxes


def iou(box0, box1):
    r0 = box0[3] / 2
    s0 = box0[:3] - r0
    e0 = box0[:3] + r0

    r1 = box1[3] / 2
    s1 = box1[:3] - r1
    e1 = box1[:3] + r1

    overlap = []
    for i in range(len(s0)):
        overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))

    intersection = overlap[0] * overlap[1] * overlap[2]
    union = box0[3] * box0[3] * box0[3] + box1[3] * box1[3] * box1[3] - intersection
    return intersection / union

def get_mask(clean_npy_path, pbb_npy_path, save_img_path = None, save_mask_path = None, affine = np.eye(4)): 
    img = np.load(clean_npy_path)
    pbb = np.load(pbb_npy_path)
    pbb = pbb[pbb[:,0]>-1]
    pbb = nms(pbb,0.05)
    mask = np.zeros(img.shape)

    for i in range(len(pbb)):
        box = pbb[i].astype('int')[1:]
        r = min(box[3], 28)  # add at 12/28
        r = max(r, 14)
        mask[0,box[0] - r: box[0] + r, box[1] - r: box[1] + r, box[2] - r: box[2] + r] = 1
    array_img = nib.Nifti1Image(img[0], affine)   # here should be modified
    array_mask = nib.Nifti1Image(mask[0], affine)
    if save_img_path != None:
        nib.save(array_img, save_img_path)
    if save_mask_path != None:
        nib.save(array_mask, save_mask_path)

        
def get_nodule_batch(clean_npy_path, pbb_npy_path, save_path = None):
    '''
    get_nodule_batch('/share5/gaor2/DSB2017-1-master/prep_test/76621363time20130228item00_clean.npy', 
                 '/share5/gaor2/DSB2017-1-master/bbox_test/76621363time20130228item00_pbb.npy', save_path = None)
    '''
    def adjust(img_shape, box, ndle_size):
        '''
        img_shape: like (1, 312, 213, 234)
        box: like [23, 45, 54]
        '''
        for i in range(3):
            box[i] = max(ndle_size//2, box[i])
            box[i] = min(box[i], img_shape[i+1] - ndle_size // 2)
        return box
    img = np.load(clean_npy_path)
    img_shape = img.shape
    print ('img size: ', img.shape)
    pbb = np.load(pbb_npy_path)
    pbb = pbb[pbb[:,0]>-1]
    pbb = nms(pbb,0.05)
    boxes = pbb[:5]
    ndle_size = 32
    nodules = np.zeros((5, ndle_size, ndle_size, ndle_size))
    boxes = sorted(boxes, key=lambda item: item[0])
    for i in range(len(boxes)):
        box = boxes[i].astype('int')[1:]
        box = adjust(img_shape, box, ndle_size)
        nodules[i,:,:,:] = img[0, box[0] - ndle_size // 2: box[0] + ndle_size//2, 
                               box[1] - ndle_size// 2: box[1] + ndle_size//2, box[2] - ndle_size//2: box[2] + ndle_size//2]
    return nodules
    
    

def get_img_mask(prep_root, bbox_root, img_root, mask_root):
    '''
    exp: get_img_mask('/share5/gaor2/DSB2017-1-master/prep_result0', '/share5/gaor2/DSB2017-1-master/bbox_result0', '/share5/gaor2/data/MCL/MCL571/img', '/share5/gaor2/data/MCL/MCL571/mask')
    '''
    prep_list = os.listdir(prep_root)
    bbox_list = os.listdir(bbox_root)
    print (len(prep_list), len(bbox_list))
    #assert len(prep_list) == len(bbox_list)
    for i in range(len(prep_list)):

        #print (prep_list[i] , bbox_list[i])
#        assert (prep_list[i][:-7] == bbox_list[i][:-7])
        if prep_list[i][-10:] != '_clean.npy':
            continue
        #print (bbox_list[i][:-10])
        name = prep_list[i][:-10]
        if i == 1:
            print (os.path.join(prep_root, name + '_clean.npy'), os.path.join(bbox_root, name + '_pbb.npy'))
            print (os.path.join(img_root, name + '.nii.gz'), os.path.join(mask_root, name + '.nii.gz'))
        if os.path.exists(os.path.join(img_root, name + '.nii.gz')):
            print (os.path.join(img_root, name + '.nii.gz'), ' already existed ========')
            continue
        #break
        print (os.path.join(img_root, name + '.nii.gz'),  ' just generated --------')
        get_mask(os.path.join(prep_root, name + '_clean.npy'), os.path.join(bbox_root, name + '_pbb.npy'), os.path.join(img_root, name + '.nii.gz'), os.path.join(mask_root, name + '.nii.gz'))
        #except:
        #    print (name + '.nii.gz with error')


def get_img_mask_condition(prep_root, bbox_root, img_root, mask_root, eld_roots):
    '''
    exp: get_img_mask('/share5/gaor2/DSB2017-1-master/prep_result0', '/share5/gaor2/DSB2017-1-master/bbox_result0', '/share5/gaor2/data/MCL/MCL571/img', '/share5/gaor2/data/MCL/MCL571/mask')
    '''
    prep_list = os.listdir(prep_root)
    bbox_list = os.listdir(bbox_root)
    print (len(prep_list), len(bbox_list))
    exclude_list = []
    for path in eld_roots:
        exclude_list += os.listdir(path)
    #assert len(prep_list) == len(bbox_list)
    for i in range(len(prep_list)):
        if prep_list[i].replace('_clean.npy', '.nii.gz') in exclude_list:
            print (prep_list[i], 'is excluded in this run')
            continue
        #print (prep_list[i] , bbox_list[i])
#        assert (prep_list[i][:-7] == bbox_list[i][:-7])
        if prep_list[i][-10:] != '_clean.npy':
            continue
        #print (bbox_list[i][:-10])
        name = prep_list[i][:-10]
        if i == 1:
            print (os.path.join(prep_root, name + '_clean.npy'), os.path.join(bbox_root, name + '_pbb.npy'))
            print (os.path.join(img_root, name + '.nii.gz'), os.path.join(mask_root, name + '.nii.gz'))
        if os.path.exists(os.path.join(img_root, name + '.nii.gz')):
            print (os.path.join(img_root, name + '.nii.gz'), ' already existed ========')
            continue
        #break
        print (os.path.join(img_root, name + '.nii.gz'),  ' just generated --------')
        get_mask(os.path.join(prep_root, name + '_clean.npy'), os.path.join(bbox_root, name + '_pbb.npy'), os.path.join(img_root, name + '.nii.gz'), os.path.join(mask_root, name + '.nii.gz'))        
        
def nodule_center(pbb_npy_path): # haven't test yet 10.23
    pbb = np.load(pbb_npy_path)
    pbb = pbb[pbb[:,0]>-1]
    pbb = nms(pbb,0.05)
    center_list = []
    for i in range(len(pbb)):
        box = pbb[i].astype('int')[1:]
        center_list.append([box[0],box[1], box[2]])
    return center_list

def mask_check(img_nii, mask_nii, pbb_npy, index, save_path = None):
    '''
    mask_check('/share5/gaor2/data/MCL/MCL571/img/6640761219time20081028item00.nii.gz', 
           '/share5/gaor2/data/MCL/MCL571/mask/6640761219time20081028item00.nii.gz', 
           '/share5/gaor2/DSB2017-1-master/bbox_result0/6640761219time20081028item00_pbb.npy', 0)
    '''
    img = nib.load(img_nii)
    img = img.get_data()
    mask = nib.load(mask_nii)
    mask = mask.get_data()
    pbb = np.load(pbb_npy)
    pbb = pbb[pbb[:,0]>-1]
    pbb = nms(pbb,0.05)
    box = pbb[index].astype('int')[1:]
    print (pbb)
    print (mask.shape)
    print (img.shape)
    ax = plt.subplot(1,2,1)
    plt.imshow(img[box[0]],'gray')          #plt.imshow(img[box[0] + 3],'gray')
    
    #plt.imshow(mask[box[0]])
    plt.axis('off')
    rect = patches.Rectangle((box[2]-box[3],box[1]-box[3]),box[3]*2,box[3]*2,linewidth=2,edgecolor='red',facecolor='none')
    ax.add_patch(rect)
    ax = plt.subplot(1,2,2)
    plt.imshow(mask[box[0]],'gray')
#plt.imshow(mask[0, box[0]])
    plt.axis('off')
    rect = patches.Rectangle((box[2]-box[3],box[1]-box[3]),box[3]*2,box[3]*2,linewidth=2,edgecolor='red',facecolor='none')
    ax.add_patch(rect)
    if save_path != None:
        ax.figure.savefig(save_path)
        ax.figure.clear()
        #3plt.clear()
    
    
def save_mask_check(norm_root, pbb_root,  save_root, item_list, gt_list = None):
    for i in range(len(item_list)):
        item = item_list[i]
        item = item.replace('.nii.gz', '')
        img_nii = norm_root + '/img/' + item + '.nii.gz'
        mask_nii = norm_root + '/mask/' + item + '.nii.gz'
        pbb_npy = pbb_root + '/DSB_File/bbox/' + item + '_pbb.npy'
        
        for index in range(3):
            save_path = save_root + '/' + item + str(index) + '.eps'
            if gt_list != None:
                save_path = save_root + '/' + item + '_index_'+str(index) + '_gt_' + str(gt_list[i])+ '.eps'
            
            mask_check(img_nii, mask_nii, pbb_npy, index, save_path)
#             except:
#                 print (item + ' error!')
            
    
def copy_preres(ori_root, new_root):
    '''
    This function copy jiachen's pre result and use my own name for DSB2017
    copy_preres('/share5/wangj46/DSB2017-1-master/prep_result0', '/share5/gaor2/DSB2017-1-master/prep_result0')
    :param ori_root:
    :param new_root:
    :return:
    '''
    item_list = os.listdir(ori_root)
    for i in range(len(item_list)):
        item_list[i] = str(item_list[i])
        item_vec = re.split('[heslaunder]', item_list[i])
        new_vec = [i for i in item_vec if len(i) != 0]
        suj = new_vec[0]
        if new_vec[0] == new_vec[1]:
            sess = new_vec[2]
        else:
            sess = new_vec[1]
        new_name = suj + 'time' + sess + 'item00'
        print ('cp ' + ori_root + '/' + item_list[i] + ' ' + new_root + '/' + new_name + item_list[i][-10:])
        os.system('cp ' + ori_root + '/' + item_list[i] + ' ' + new_root + '/' + new_name + item_list[i][-10:])
        #print ('cp ' + ori_root + '/' + item_list[i] + ' ' + new_root + '/' + new_name + item_list[i][-8:])

def copy_predata(ori_root, new_root):
    '''
    This function copy jiachen's pre result and use my own name for /img /mask
    exp: copy_predata('/share5/wangj46/fromlocal/318box', '/share5/gaor2/data/MCL/MCL277')
    :param ori_root:
    :param new_root:
    :return:
    '''
    img_list = os.listdir(ori_root + '/img')
    mask_list = os.listdir(ori_root + '/mask')
    assert len(img_list) == len(mask_list)
    print ('the number to be copied', len(img_list))
    for i in range(len(img_list)):
        assert img_list[i] == mask_list[i]
        img_list[i] = str(img_list[i])
        item_vec = re.split('[heslaunder_-]', img_list[i])
        new_vec = [i for i in item_vec if len(i) != 0]
        suj = new_vec[0]
        if new_vec[0] == new_vec[1]:
            sess = new_vec[2]
        else:
            sess = new_vec[1]
        new_name = suj + 'time' + sess + 'item00.nii.gz'
        print ('cp ' + ori_root + '/img/' + img_list[i] + ' ' + new_root + '/img/' + new_name)
        print ('cp ' + ori_root + '/mask/' + mask_list[i] + ' ' + new_root + '/mask/' + new_name)
        os.system('cp ' + ori_root + '/mask/' + mask_list[i] + ' ' + new_root + '/mask/' + new_name)
        os.system('cp ' + ori_root + '/img/' + img_list[i] + ' ' + new_root + '/img/' + new_name)

def get_trval_table(img_root, save_tab_root):
    '''
    exp: get_trval_table('/share5/gaor2/data/MCL/MCL571/img', '/share5/gaor2/data/MCL/MCL571/labels.csv')
    :param img_root:
    :param save_tab_root:
    :return:
    '''
    img_list = os.listdir(img_root)
    data = pd.DataFrame()
    subj_col = []
    sess_col = []
    path_col = []
    data['item'] = img_list
    tr_val = []
    for i in range(len(img_list)):

        name_vec = re.split('[rgtime.n]', img_list[i])
        new_vec = [i for i in name_vec if len(i) != 0]
        subj_name = new_vec[0]
        sess_name = new_vec[1]
        print (subj_name, sess_name)
        path_name = os.path.join(img_root, img_list[i])
        subj_col.append(subj_name)
        sess_col.append(sess_name)
        path_col.append(path_name)
#        tr_val.append( i % 5)                   # this cannot apply to RNN based method
    data['subject'] = subj_col
    data['session'] = sess_col
    data['path'] = path_col
#    data['trainvalnew'] = tr_val
    data.to_csv(save_tab_root)
    
def get_trval_col(ori_csv,new_csv):
    '''
    This function create the 5 splits based on subjects
    '''
    df = pd.read_csv(ori_csv)
    id_list = df['subject']
    id_set = set(id_list)
    new_id_list = list(id_set)
    trval_list = []
    for i in range(len(id_list)):
        index = new_id_list.index(id_list[i])
        trval_list.append(index % 5)
    df['trainvalnew'] = trval_list
    df.to_csv(new_csv)
    
def id_label_NLST(cancer_csv, nocancer_csv):
    can_df = pd.read_csv(cancer_csv)
    can_list = can_df['Demographics.pid']
    nocan_df = pd.read_csv(nocancer_csv)
    nocan_list = nocan_df['Demographics.pid']
    id_label = {}
    for label in can_list:
        id_label[str(label)] = 1
    for label in nocan_list:
        id_label[str(label)] = 0
    return id_label

def id_label_dict(MCL_csvs, MCL_xlsx):
    id_label = {'0': [], '1': [], '-1': []}
    MCLdf0 = pd.read_csv(MCL_csvs[1])
    for i, item in MCLdf0.iterrows():
        if item['Histologic Type'] != item['Histologic Type']:
            continue
        if item['Histologic Type'].strip() in ['Adenocarcinoma', 'Large Cell Neuroendocrine', 'Non Small Cell (NSCLC)', 'Small Cell Carcinoma', 'Squamous Cell Carcinoma']:
            id_label['1'].append(str(item['Lookup MCL']))
        if item['Histologic Type'].strip() in ['Adenoid Cystic Carcinoma', 'Adenosquamous Carcinoma', 'Atypical Carcinoid', 'Carcinoid', 'Stage IB', 'Stage IIB', 'other', 'Other', 'IGNORE', 'No Diagnosis']:
            id_label['-1'].append(str(item['Lookup MCL']))
        if item['Histologic Type'].strip() in ['Granuloma', 'Negative for Dysplasia and Metaplasia', 'Negative for Malignant Cells', 'Squamous Metaplasia', 'Normal']:
            id_label['0'].append(str(item['Lookup MCL']))
    
    MCLdf1 = pd.read_csv(MCL_csvs[0])
    for i, item in MCLdf1.iterrows():
        if item['Histologic.Type'] != item['Histologic.Type']:
            continue
        if item['Histologic.Type'].strip() in ['Adenocarcinoma', 'Large Cell Neuroendocrine', 'Non Small Cell (NSCLC)', 'Small Cell Carcinoma', 'Squamous Cell Carcinoma']:
            id_label['1'].append(str(item['MCL.id']))
        if item['Histologic.Type'].strip() in ['Adenoid Cystic Carcinoma', 'Adenosquamous Carcinoma', 'Atypical Carcinoid', 'Carcinoid', 'Stage IB', 'Stage IIB', 'other', 'Other', 'IGNORE', 'No Diagnosis']:
            id_label['-1'].append(str(item['MCL.id']))
        if item['Histologic.Type'].strip() in ['Granuloma', 'Negative for Dysplasia and Metaplasia', 'Negative for Malignant Cells', 'Squamous Metaplasia', 'Normal' ]:
            id_label['0'].append((item['MCL.id']))
        
    df = pd.read_excel(MCL_xlsx, sheet_name = 'update.all.dat1')   
    for i, item in df.iterrows():
        if item['Histologic.Type'] != item['Histologic.Type']:
            continue
        if item['Histologic.Type'].strip() in ['Adenocarcinoma', 'Large Cell Neuroendocrine', 'Non Small Cell (NSCLC)', 'Small Cell Carcinoma', 'Squamous Cell Carcinoma']:
            id_label['1'].append(str(item['Image.ID']))
        if item['Histologic.Type'].strip() in ['Adenoid Cystic Carcinoma', 'Adenosquamous Carcinoma', 'Atypical Carcinoid', 'Carcinoid', 'Stage IB', 'Stage IIB', 'other', 'Other', 'IGNORE', 'No Diagnosis']:
            id_label['-1'].append(str(item['Image.ID']))
        if item['Histologic.Type'].strip() in ['Granuloma', 'Negative for Dysplasia and Metaplasia', 'Negative for Malignant Cells', 'Squamous Metaplasia', 'Normal' ]:
            id_label['0'].append((item['Image.ID']))
    
    return id_label
    


def add_col(ori_csv,new_csv, id_label):
    '''
    id_label = id_label_dict('/share5/gaor2/data/MCL/MCL_csv/MCL_match_070618.csv')
    add_col('/share5/gaor2/data/MCL/MCL571/labels.csv','/share5/gaor2/data/MCL/MCL571/labels1.csv', id_label)
    '''
    df = pd.read_csv(ori_csv)
    id_list = df['subject']
   
    label_list = []
    for i in range(len(id_list)):
        if str(id_list[i]) in id_label.keys():
            label_list.append(id_label[str(id_list[i])])
        else:
            label_list.append('')
    df['label'] = label_list
    df.to_csv(new_csv)

def delete_row(ori_csv, new_csv, item_name):
    '''
    delete the row which doesn't have a label. 
    delete_row('/share5/gaor2/data/MCL/MCL571/labels1.csv', '/share5/gaor2/data/MCL/MCL571/labels2.csv')
    '''
    df = pd.read_csv(ori_csv)
    label_list = df[item_name]
    print (label_list)
    label_array = np.array(label_list)
    print (len(df))
    nll_index = np.where(label_array != label_array)[0]
    print (nll_index)
    new_df = df.drop(nll_index)
    print (len(new_df))
    new_df.to_csv(new_csv)

def xnat_path_sess(path_txt, sess_txt):
    '''
    Get a txt file of sesses for a txt file of pathes. 
    exp: xnat_path_sess('/share5/gaor2/data/MCL/txt/Xnat/MCL_sessions_achived0926.txt',
               '/share5/gaor2/data/MCL/txt/Xnat/MCL_subj0926.txt')
    '''
    f = open(path_txt, 'r')
    f_s = open(sess_txt, 'w')
    lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i]
        line_vec = line.strip().split('/')
        f_s.write(line_vec[-2] + '\n')
    f_s.close()
    f.close()

def get_QA_table(data_root, save_root):
    '''
    create the table with subject session and item for efficient to add something. 
    get_QA_table('/share5/gaor2/data/MCL_data2', '/share5/gaor2/data/MCL_data2_QA.csv')
    :param data_root:
    :param save_root:
    :return:
    '''
    subj_list = os.listdir(data_root)
    data = pd.DataFrame()
    subj_col = []
    sess_col = []
    item_col = []
    for i in range(len(subj_list)):
        print (i)
        subj_path = data_root + '/' + subj_list[i]
        sess_list = os.listdir(subj_path)
        for j in range(len(sess_list)):
            sess_path = subj_path + '/'+ sess_list[j]
#             subj_col.append(subj_list[i])
#             sess_col.append(sess_list[j])
#             item_col.append( subj_list[i] + 'time' + sess_list[j] + '.nii.gz')
            item_list = os.listdir(sess_path)
            for k in range(len(item_list)):
                subj_col.append(subj_list[i])
                sess_col.append(sess_list[j])
                item_col.append(item_list[k])
    data['subject name'] = subj_col
    data['time'] = sess_col
    data['nifti name'] = item_col
    data.to_csv(save_root)

def get_regQA_table(data_root, save_root):
    sess_list = os.listdir(data_root)
    data = pd.DataFrame()
    data['session'] = sess_list
    data.to_csv(save_root)
    
def add_to_combine(combine_path, newdata_path, log_path):
    """
    This data aims to add the new data to combine_path, combine_path means the total images available in MCL. 
    exp: add_to_combine('/share5/gaor2/data/MCL/MCLnorm/combine', '/share5/gaor2/data/MCL/MCL277', '/share5/gaor2/data/MCL/MCLnorm/copylog/mcl227toCombine.txt')
    """
    f = open(log_path, 'w')
    f.write('the log of copy '+ newdata_path + ' to ' + combine_path + '\n')
    img_list = os.listdir(newdata_path + '/img')
    mask_list = os.listdir(newdata_path + '/mask')
    assert (len(img_list) == len(mask_list))
    
    for i in range(len(img_list)):
        print (img_list[i])
        if os.path.exists(combine_path + '/img/' + img_list[i]):
            print (combine_path + '/img/' + img_list[i] + ' is existed')
            f.write(combine_path + '/img/' + img_list[i] + ' is existed\n')
            continue
        os.system('cp ' + newdata_path + '/img/' + img_list[i] + ' ' + combine_path + '/img/')
        os.system('cp ' + newdata_path + '/mask/' + img_list[i] + ' ' + combine_path + '/mask/')
        
def folder_according_sub(ori_path, new_path):
    '''
    Because if we make the folder according to sessions, some subjects will have sessions both in training set and test set. 
    So, we change the floder according to subjects
    exp: folder_according_sub('/share5/gaor2/data/MCL/MCL571/labels3.csv', '/share5/gaor2/data/MCL/MCL571/labels4.csv')
    '''
    ori_df = pd.read_csv(ori_path)
    val_list, test_list = [], []
    for i, item in ori_df.iterrows():
        if item['trainvalnew'] == 4:
            val_list.append(item['subject'])
        if item['trainvalnew'] == 0:
            test_list.append(item['subject'])
    new_trainval = []
    tr_cnt, val_cnt, tt_cnt = 0,0,0
    for i, item in ori_df.iterrows():
        if item['subject'] in val_list:
            new_trainval.append(4)
            val_cnt += 1
        elif item['subject'] in test_list:
            new_trainval.append(0)
            tt_cnt += 1
        else:
            new_trainval.append(1)
            tr_cnt += 1
    ori_df['trainvalnew'] = new_trainval
    print (val_cnt, tt_cnt, tr_cnt)
    ori_df.to_csv(new_path)
            
def find_miss_sub(ori_csv_path, data_folders):
    '''
    To find if there is some subjects that appears in csv and not in data folders
    exp: find_miss_sub('/share5/gaor2/data/MCL/MCL_csv/MCL Nodule_2018OCT9.csv', 
              ['/share5/gaor2/data/MCL/MCL_time/MCL0706', '/share5/gaor2/data/MCL/MCL_time/MCL0826', '/share5/gaor2/data/MCL/MCL_time/MCL0706'])
    '''
    df = pd.read_csv(ori_csv_path)
    mclid_list = df['MCL.ID'].tolist()
    mclid_set = set(mclid_list)
    subject_set = set([])
    for i in range(len(data_folders)):
        folder = data_folders[i]
        sub_list = os.listdir(folder)
        subject_set.update(sub_list)
    print (mclid_set - subject_set)
    print (subject_set - mclid_set)
        
def find_miss_sub2(ori_csv_path, subject_txt):
    '''
    To find if there is some subjects that appears in csv and not in data folders. 
    find_miss_sub2('/share5/gaor2/data/MCL/MCL_csv/MCL Nodule_2018OCT9.csv', '/share5/gaor2/data/MCL/txt/Xnat/final_subject_list.txt')
    '''
    df = pd.read_csv(ori_csv_path)
    mclid_list = df['MCL.ID'].tolist()
    mclid_set = set(mclid_list)
    f = open(subject_txt)
    subject_set = set([])
    s_lines = f.readlines()
    for i in range(len(s_lines)):
        line_vec = re.split('[-_]', s_lines[i].strip())
        subject_set.add(line_vec[0])
    print (subject_set)
    print (len(subject_set))
    data1, data2 = pd.DataFrame(), pd.DataFrame()
    m_s_set = mclid_set - subject_set
    s_m_set = subject_set - mclid_set
    s_m, m_s = [], []
    for n in m_s_set: 
        if n != '':
            m_s.append(n)
    for n in s_m_set: 
        if n != '':
            s_m.append(n)
            
    data1['in xnat but not in csv'] = s_m
    data2['in csv but not in xnat'] = m_s
    data1.to_csv('xnat-csv.csv')
    data2.to_csv('csv-xnat.csv')
#     print (mclid_set - subject_set)
#     print (subject_set - mclid_set)

def change_path_csv(csv_path):
    df = pd.read_csv(csv_path)
    path_list = df['path'].tolist()
    new_path_list = []
    for i in range(len(path_list)):
        print ('ori path: ', path_list[i])
        tmp_path = path_list[i].replace( 'MCL277', 'MCLnorm/MCL277')
        print ('new_path: ', tmp_path)
        new_path_list.append(tmp_path)
    df['path'] = new_path_list
    df.to_csv(csv_path)
    
def get_subj_name(data_root):   # maybe only use once, just for xnat1025
    subj_set = set([])
    for folder_name in ['L_3', 'L_4', 'UPDATE_L3_09_07_2018', 'UPDATE_L4_09_07_2018']:
        sub_list = os.listdir(os.path.join(data_root, folder_name))
        for subj in sub_list:
#            print (subj)
            if len(subj) == 10 or len(subj) == 11 or len(subj) == 9:
                subj_set.add(subj)
    return subj_set

def get_subj_name2(txt_path):
    subj_set = set([])
    f = open(txt_path)
    sess_list = f.readlines()
    for sess in sess_list:
        subj = sess.split('_')[0]
        subj_set.add(subj)
    return subj_set
    
def cnt_kaggle_nodule(data_root):
    f_list = os.listdir(data_root)
    cnt_all, cnt_nod =0, 0
    for i in range(len(f_list)):
        if f_list[i][-8:] != '_pbb.npy':
            continue
        pbb = np.load(data_root + '/' + f_list[i])
        pbb = pbb[pbb[:, 0] > -1]
        pbb = nms(pbb, 0.05)
        cnt_all += 1
        if len(pbb) != 0:
            cnt_nod += 1
        print (cnt_all, cnt_nod)
        
def cnt_kaggle_label(csv_path, data_root):   # this one has been used for SPIE paper. 
    data_list = os.listdir(data_root)
    new_list = [i[:-7] for i in data_list]
    df = pd.read_csv(csv_path)
    img_list = os.listdir()
    cnt_all, cnt_pos = 0, 0
    for i, item in df.iterrows():
        if item['id'] in new_list:
            cnt_all += 1
            cnt_pos += item['cancer']
        
    print (cnt_all, cnt_pos)
    
def cnt_MCL_subject(data_root):
    '''
    give all the sessions in a folder, return a dictory with id and session list
    cnt_MCL_subject('/share5/gaor2/data/MCL/MCLnorm/MCL571/img')
    '''
    data_list = os.listdir(data_root)
    dct = {}
    for i in range(len(data_list)):
        subj_id = re.split('[item]', data_list[i].strip())[0]
        print (subj_id)
        if subj_id not in dct.keys():
            dct[subj_id] = []
        dct[subj_id].append(data_list[i])
    cnt = 0
    for key in dct.keys():
        if len(dct[key]) > 1:
            cnt += 1
    print (cnt)

def find_avaible_subj(ori_clinical_path, MCL_time_path, save_path):
    '''
    This function original for creating available subject list in MCL data. Because we need to add more data MCL to training and test. 
    '''
    df_ori = pd.read_csv(ori_clinical_path)
    mcl_list = []
    time_list = os.listdir(MCL_time_path)
    for i in range(len(time_list)):
        tmp_mcl_list = os.listdir(os.path.join(MCL_time_path, time_list[i]))
        mcl_list += tmp_mcl_list
    drop_list = []
    print (len(df_ori))
    for i, item in df_ori.iterrows():
        if item['MCL.ID'] not in mcl_list:
            drop_list.append(i)
    print ('drop_list is: ', drop_list)
    print ('len of drop_list: ', len(drop_list))
    df_new = df_ori.drop(drop_list)
    df_new.to_csv(save_path)
    
def combine_mcl_time(data_root, sub_roots):
    '''
    To create a combine dir for MCL_time
    '''
    cmb_root = os.path.join(data_root, 'combine')
    cmb_list = os.listdir(cmb_root)
    for sub in sub_roots:
        sub_root = os.path.join(data_root, sub)
        sub_list = os.listdir(sub_root)
        for i in range(len(sub_list)):
            sub_subj_root = os.path.join(sub_root, sub_list[i])
            if sub_list not in cmb_list:
                print ('copy all subjects')
                print ('cp -r ' + sub_subj_root + ' ' + cmb_root)
                os.system('cp -r ' + sub_subj_root + ' ' + cmb_root)
            else:
                sub_sess_list = os.listdir(sub_subj_root)
                cmb_sess_list = os.listdir(os.path.join(cmb_root, sub_list[i]))
                for j in range(len(sub_sess_list)):
                    if sub_sess_list[j] not in cmb_sess_list:
                        print ('copy session')
                        print ('cp -r ' + os.path.join(sub_subj_root, sub_sess_list[j]) + ' ' + os.path.join(cmb_root, sub_list[i]))
                        os.system('cp -r ' + os.path.join(sub_subj_root, sub_sess_list[j]) + ' ' + os.path.join(cmb_root, sub_list[i]))
                
def correct_sess_time(data_root):  # this is only for MCL1025 ,have not test yet
    subj_list = os.listdir(data_root)
    for i in range(len(subj_list)):
        subj_path = os.path.join(data_root, subj_list[i])
        sess_list = os.listdir(subj_path)
        for j in range(len(sess_list)):
            sess_name = sess_list[j]
#            sess_path = os.path.join(subj_path, sess_path)
            if len(sess_name) != 8:
                print (subj_list[i], sess_name, 'len is not enough =====================')
                continue
            time_str = sess_name
#             if int(time_str[0:4]) < int(time_str[4:8]):
#                 new_time = time_str[4:8] + time_str[0:4]
#                 new_sess_name = sess_name.replace(time_str, new_time)
#                 print ('mv ' + os.path.join(subj_path, sess_name) + ' ' + os.path.join(subj_path, new_time))
#                 os.system('mv ' + os.path.join(subj_path, sess_name) + ' ' + os.path.join(subj_path, new_sess_name))
            
#             if len(sess_name) < 34:
#                 print (sess_name, 'len is not enough')
#                 continue
#                 time_str = sess_name[-15:-7]
#                 if int(time_str[0:4]) < int(time_str[4:8]):
#                     new_time = time_str[4:8] + time_str[0:4]
#                     new_sess_name = sess_name.replace(time_str, new_time)
#                     print ('mv ' + os.path.join(subj_path, sess_name) + ' ' + os.path.join(subj_path, new_sess_name))
#                    os.system('mv ' + os.path.join(subj_path, sess_name) + ' ' + os.path.join(subj_path, new_sess_name))

def correct_sess_time2(data_root):  # this is only for MCL1025 
    subj_list = os.listdir(data_root)
    for i in range(len(subj_list)):
        subj_path = os.path.join(data_root, subj_list[i])
        sess_list = os.listdir(subj_path)
        for j in range(len(sess_list)):
            sess_name = sess_list[j]
            sess_path = os.path.join(subj_path, sess_name)
            item_list = os.listdir(sess_path)
            if (len(item_list) > 1): print ('========== len(item_list) > 1', subj_list[i], sess_name)
            print ('mv ' + os.path.join(sess_path, item_list[0]) + ' ' + os.path.join(sess_path, subj_list[i]+ 'time' + sess_name + 'item00.nii.gz'))
            os.system('mv ' + os.path.join(sess_path, item_list[0]) + ' ' + os.path.join(sess_path, subj_list[i]+ 'time' + sess_name + 'item00.nii.gz'))


def cnt_max_resolution(data_root):
    '''
    dim_list = cnt_max_resolution('/share3/gaor2/share5backup/data/MCL/MCL_time/MCL0826')
    '''
    subj_list = os.listdir(data_root)
    dim_list = []
    for i in range(len(subj_list)):
        subj_path = os.path.join(data_root, subj_list[i])
        sess_list = os.listdir(subj_path)
        for j in range(len(sess_list)):
            sess_name = sess_list[j]
            sess_path = os.path.join(subj_path, sess_name)
            item_list = os.listdir(sess_path)
            img = nib.load(os.path.join(sess_path, item_list[0]))
            x_dim = img.header['dim'][1] * img.header['pixdim'][1] 
            y_dim = img.header['dim'][2] * img.header['pixdim'][2] 
            z_dim = img.header['dim'][3] * img.header['pixdim'][3] 
            dim_list.append([x_dim, y_dim, z_dim])
    return dim_list
            
def rm_small(data_root):   # after modified, not test
    '''
    This function only maintain the largest file in a folder 
    rm_small('/share3/gaor2/share5backup/data/MCL/MCL_time/combine/366660798/20140220')
    '''
    data_list = glob(os.path.join(data_root, '*.nii.gz'))
    size_list = []
    for i in range(len(data_list)):
        item = data_list[i]
        size_list.append(os.path.getsize(item))
    max_index = size_list.index(max(size_list))

    for i in range(len(data_list)):
        if i != max_index:
            os.system('rm ' +  data_list[i])
            print ('rm ' +  data_list[i])

def rm_multi_item(data_root, log_txt = None):
    '''
    This function work with rm_small(), remove the multi items in a dataset
    exp: rm_multi_item('/share3/gaor2/share5backup/data/MCL/MCL_time/combine', '/share3/gaor2/share5backup/data/MCL/MCL_time/combine2.txt')
    '''
    subj_list = os.listdir(data_root)
#    f = open(log_txt, 'w')
#    f.write('These sessions have something deleted!\n')
    for i in range(len(subj_list)):
        subj_path = os.path.join(data_root, subj_list[i])
        sess_list = os.listdir(subj_path)
        for j in range(len(sess_list)):
            sess_name = sess_list[j]
            sess_path = os.path.join(subj_path, sess_name)
            if len(os.listdir(sess_path)) > 1:
#                f.write(sess_path + '\n')
                print (sess_path)
                rm_small(sess_path)
        
#    f.close()
    
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
                
def find_small_item(data_root, th_size, log_txt=None):
    subj_list = os.listdir(data_root)
    small_list = []
    #f = open(log_txt, 'w')
    #f.write('These sessions are small than 15 M deleted!\n')
    for i in range(len(subj_list)):
        subj_path = os.path.join(data_root, subj_list[i])
        sess_list = os.listdir(subj_path)
        for j in range(len(sess_list)):
            sess_name = sess_list[j]
            sess_path = os.path.join(subj_path, sess_name)
#             if len(os.listdir(sess_path)) == 0:
#                 os.system('rm -r ' + sess_path)
            item = sess_path + '/' + os.listdir(sess_path)[0]
            if os.path.getsize(item) < th_size:
                small_list.append(item)
                print (item)
    return small_list 
    #f.close()


    
def delete_wholebody(data_root, csv_path): # this works for jiachen QA table
    df = pd.read_csv(csv_path)
    for i, item in df.iterrows():
        if item['qa'] == 'wholebody':
            name_vec = re.split('[_-]', item['session'])
            subj_name = name_vec[0]
            sess_name = name_vec[-1]
            if not os.path.exists(os.path.join(data_root, subj_name, sess_name)):
                print (os.path.join(data_root, subj_name, sess_name) + 'has been removed')
                continue
            if (len(os.listdir(data_root + '/' + subj_name)) == 1): 
                os.system('rm -r ' + data_root + '/' + subj_name)
                print ('rm -r ' + data_root + '/' + subj_name)
            else:
                os.system('rm -r ' + os.path.join(data_root, subj_name, sess_name))
                print ('rm -r ' + os.path.join(data_root, subj_name, sess_name))
        

def delete_empty_NLST(data_root):
    subj_list = os.listdir(data_root)
    for i in range(len(subj_list)):
        subj_path = os.path.join(data_root, subj_list[i])
        sess_list = os.listdir(subj_path)
        for j in range(len(sess_list)):
            sess_path = os.path.join(subj_path, sess_list[j])
            if len(glob(os.path.join(sess_path, '*.nii.gz'))) == 0:
                print ('rm -r ', sess_path)
                os.system('rm -r ' + sess_path)

def cp_reg_spread(data_root, new_root, QA_csv):
    '''
    cp_reg_spread('/share3/gaor2/share5backup/data/MCL/Registration/regfinal', '/share3/gaor2/share5backup/data/MCL/MCLspread/Regcombine', '/share3/gaor2/share5backup/data/MCL/Registration/QA/1119_QA.csv')
    '''
    df = pd.read_csv(QA_csv)
    for i, item in df.iterrows():
        if item['status'] == item['status']:
            continue
        mkdir(os.path.join(new_root, item['session']))
        print ('cp ' + os.path.join(data_root, item['session'], 'home-nfs_deformed.nii.gz') + 
                 ' ' + os.path.join(new_root, item['session'], item['session'] + '.nii.gz'))
        os.system('cp ' + os.path.join(data_root, item['session'], 'home-nfs_deformed.nii.gz') + 
                 ' ' + os.path.join(new_root, item['session'], item['session'] + '.nii.gz'))
        
def cp_reg_spread_NLST(data_root, new_root):
    '''

    '''
    item_list = os.listdir(data_root)
    for i in range(len(item_list)):
        if not os.path.exists(data_root + '/' + item_list[i] + '/home-nfs_deformed.nii.gz'):
            print (data_root + '/' + item_list[i] + '/home-nfs_deformed.nii.gz is not existed')
            continue
        mkdir(new_root + '/' + item_list[i])
        if os.path.exists(os.path.join(new_root, item_list[i], item_list[i] + '.nii.gz')):
            print (os.path.join(new_root, item_list[i], item_list[i] + '.nii.gz') + 'is existed==')
            continue
        print ('cp ' + os.path.join(data_root, item_list[i], 'home-nfs_deformed.nii.gz') + 
                 ' ' + os.path.join(new_root, item_list[i], item_list[i] + '.nii.gz'))
        os.system('cp ' + os.path.join(data_root, item_list[i], 'home-nfs_deformed.nii.gz') + 
                 ' ' + os.path.join(new_root, item_list[i], item_list[i] + '.nii.gz'))
        
def cp_ref_spread_NLST(data_root, new_root, pair_csv):
    item_list = os.listdir(data_root)
    df = pd.read_csv(pair_csv)
    sub_refsess_dict = {}
    for i,item in df.iterrows():
        sub_refsess_dict[str(item['subject'])] = item['ref_item'][6:10] # ref_item like 01-02-2001-.....
        
    for i in range(len(item_list)):
        print (i)
        if not os.path.exists(data_root + '/' + item_list[i] + '/ref_padding.nii.gz'):
            print (data_root + '/' + item_list[i] + '/ref_padding.nii.gz is not existed')
            continue
        
        subj_name = item_list[i][3:9]  # item: like reg200001time2000
        new_name = 'reg' + subj_name + 'time' + sub_refsess_dict[subj_name]
        
        if os.path.exists(os.path.join(new_root, new_name, new_name + '.nii.gz')):
            print (os.path.join(new_root, new_name, new_name + '.nii.gz') + 'is existed==')
            continue
        mkdir(os.path.join(new_root, new_name))
        print ('cp ' + os.path.join(data_root, item_list[i].replace('reg', 'tmp'), 'ref_padding.nii.gz') + 
                 ' ' + os.path.join(new_root, new_name, new_name + '.nii.gz'))
        os.system('cp ' + os.path.join(data_root, item_list[i].replace('reg', 'tmp'), 'ref_padding.nii.gz') + 
                 ' ' + os.path.join(new_root, new_name, new_name + '.nii.gz'))
        
def cp_ref_spread(ori_root, data_root, new_root, QA_csv):
    '''
    cp_ref_spread('/share3/gaor2/share5backup/data/MCL/MCL_time/combine', '/share3/gaor2/share5backup/data/MCL/Registration/regtmp', '/share3/gaor2/share5backup/data/MCL/MCLspread/Regcombine', '/share3/gaor2/share5backup/data/MCL/Registration/QA/1119_QA.csv')
    '''
    df = pd.read_csv(QA_csv)
    subj_ref_dict = {}
    ori_subject_list = os.listdir(ori_root)
    for i in range(len(ori_subject_list)):
        subj_name = ori_subject_list[i]
        print (subj_name)
        if (len(os.listdir(ori_root + '/' + subj_name)) == 0):
            continue
        sess_name = max(os.listdir(ori_root + '/' + subj_name))
        subj_ref_dict[str(subj_name)] = str(sess_name)
        
    for i, item in df.iterrows():
        if item['status'] == item['status']:
            continue
        subj_name = re.split('[gt]', item['session'])[1]
        new_name = 'reg' + subj_name + 'time' + subj_ref_dict[subj_name]
        
        mkdir(os.path.join(new_root, new_name))
        if (len(os.listdir(os.path.join(new_root, new_name))) != 0):
            continue
        print ('cp ' + os.path.join(data_root, item['session'].replace('reg', 'tmp'), 'ref_padding.nii.gz') + 
                 ' ' + os.path.join(new_root, new_name, new_name + '.nii.gz'))
        os.system('cp ' + os.path.join(data_root, item['session'].replace('reg', 'tmp'), 'ref_padding.nii.gz') + 
                 ' ' + os.path.join(new_root, new_name, new_name + '.nii.gz'))


        
def cp_single_sess(ori_root, tmp_root, new_root):
    '''
    There are three steps: cp_reg_spread, cp_ref_spread, cp_single_sess 
    
    cp_single_sess('/share3/gaor2/share5backup/data/MCL/MCL_time/combine', '/share3/gaor2/share5backup/data/MCL/Registration/single_session', '/share3/gaor2/share5backup/data/MCL/MCLspread/Regcombine')
    '''
    subject_list = os.listdir(ori_root)
    for i in range(len(subject_list)):
        print ('==============', i, subject_list[i])
        subj_name = subject_list[i]
        if (len(os.listdir(ori_root + '/' + subj_name)) != 1):
            continue
        sess_name = max(os.listdir(ori_root + '/' + subj_name))
        item_paths = glob(os.path.join(ori_root, subj_name, sess_name, '*.nii.gz'))
        if len(item_paths) != 1:

            print ('item_path', item_paths)
            continue
        item_path = item_paths[0]
        tmp_path = os.path.join(tmp_root, 'tmp' + subj_name + 'time' + sess_name)
        mkdir(tmp_path)
        if len(os.listdir(tmp_path)) != 4:
            print ('len(os.listdir(tmp_path)) != 4')
            continue
#         print ('python /home/local/VANDERBILT/gaor2/code/RNNLung/func/tools/seg_lung.py --ori ' + item_path + ' --out ' + tmp_path + '/ref_mask.nii.gz')
#         os.system('python /home/local/VANDERBILT/gaor2/code/RNNLung/func/tools/seg_lung.py --ori ' + item_path + ' --out ' + tmp_path + '/ref_mask.nii.gz')
#         os.system('python /home/local/VANDERBILT/gaor2/code/RNNLung/func/tools/get_new_lung.py --ori '+ item_path + ' --mask ' + tmp_path + '/ref_mask.nii.gz ' +'--out ' + tmp_path + '/ref_seg.nii.gz')
#         os.system('/fs4/masi/huoy1/Software/freesurfer6/freesurfer/bin/mri_convert ' + tmp_path + '/ref_seg.nii.gz ' + tmp_path + '/ref_resample.nii.gz -vs 1.0 1.0 1.0')
#         os.system('python /home/local/VANDERBILT/gaor2/code/RNNLung/func/tools/padding.py --ori '+ tmp_path + '/ref_resample.nii.gz --out ' + tmp_path + '/ref_padding.nii.gz')
        new_name = 'reg' + subj_name + 'time' + sess_name
        mkdir(os.path.join(new_root, new_name))
        print ('cp ' + tmp_path + '/ref_padding.nii.gz ' + os.path.join(new_root, new_name, new_name + '.nii.gz'))
        os.system('cp ' + tmp_path + '/ref_padding.nii.gz ' + os.path.join(new_root, new_name, new_name + '.nii.gz'))
        
def cp_save_max_NLST(ori_root, new_root):
    '''
    This take the NLST and 
    '''
    subj_list = os.listdir(ori_root)
    for i in range(len(subj_list)):
        print (i, subj_list[i])
#        if i > 1: break
        subj_path = os.path.join(ori_root, subj_list[i])
        sess_list = os.listdir(subj_path)
        for j in range(len(sess_list)):
            data_list = glob(os.path.join(subj_path + '/' + sess_list[j], '*.nii.gz'))
            if len(data_list) == 0:
                print (subj_path + '/' + sess_list[j], ' has no nifti')
            size_list = []
            for k in range(len(data_list)):
                item = data_list[k]
                size_list.append(os.path.getsize(item))
            max_index = size_list.index(max(size_list))
            new_path = new_root + '/' + subj_list[i] + '/' + sess_list[j]
            mkdir(new_path)
            if (len(os.listdir(new_path)) != 0):
                print (new_path, ' is existed')
                continue
            print ('cp ' + data_list[max_index] + ' ' + new_path)
            os.system('cp ' + data_list[max_index] + ' ' + new_path)
        
def debug_dicom(dicom_root):
    '''
    This function is to see what slice has been lost in a dicom folder
    '''
    dcm_list = os.listdir(dcm_root)
    num_list = []
    for i in range(1, len(dcm_list)):
        ds = pydicom.dcmread(dcm_root + '/' + dcm_list[i])
        num_list.append(int(ds[0x20, 0x13].value))
    for i in range(max(num_list)):
        if i not in num_list:
            print (i)

def cnt_SPORE_info0(csv_path, data_root, save_csv_path):
    '''
    This function compare the spore data on the dataset and spreadsheet. 
    '''
    dict_xnat = {}
    dict_sheet = {}
    df = pd.read_csv(csv_path)
    subj_list = os.listdir(data_root)
    for i in range(len(subj_list)):
        if subj_list[i] not in dict_xnat.keys():
            dict_xnat[subj_list[i]] = 0
        dict_xnat[subj_list[i]] += 1
    sheet_list = pd.read_csv(csv_path)['sub_name'].tolist()
    sheet_list = [str(i) for i in sheet_list]
    for i in range(len(sheet_list)):
        if sheet_list[i] not in dict_sheet.keys():
            dict_sheet[sheet_list[i]] = 0
        dict_sheet[sheet_list[i]] += 1
    
    inxnat = []
    cnt1 = 0
    cnt2 = 0
    for i, item in df.iterrows():
        if item['sub_name'] not in subj_list:
            inxnat.append(0)
        else:
            if dict_xnat[item['sub_name']] == dict_sheet[item['sub_name']]:
                inxnat.append(1)
                cnt1 += 1
            if dict_xnat[item['sub_name']] < dict_sheet[item['sub_name']]:
                inxnat.append(2)
                cnt2 += 1
            if dict_xnat[item['sub_name']] > dict_sheet[item['sub_name']]:
                print ('========error===', item['sub_name'])
                
    df['If on Xnat (no=0, yes=1)'] = inxnat
    print (cnt1)
    print (cnt2)
    print (sum(inxnat))
#    df.to_csv(save_csv_path)

def cnt_SPORE_info(csv_path, data_root, save_csv_path):
    '''
    This function compare the spore data on the dataset and spreadsheet. 
    '''
    dict_xnat = {}
    df = pd.read_csv(csv_path)
    subj_list = os.listdir(data_root)
    cnt = 0
    for i in range(len(subj_list)):
        dict_xnat[subj_list[i]] = os.listdir(data_root + '/' + subj_list[i])
        sess_list = os.listdir(data_root + '/' + subj_list[i])
        for j in range(len(sess_list)):
            if sess_list[j][:4] != '2018':
                cnt += 1
                print (subj_list[i], sess_list[j])

    print (cnt , '===')
    inxnat = []
    
    for i, item in df.iterrows():
        spore_id = item['sub_name'][6:]
        if spore_id not in subj_list:
            inxnat.append(0)
        else:
            date_vec = item['studydate'].split('-')
            #print (item['sub_name'],item['studydate'])
            #date = '20' + date_vec[2] + date_vec[0].zfill(2) + date_vec[1].zfill(2)
            date = str(date_vec[0]) + str(date_vec[1]) + str(date_vec[2])
            #if str(date_vec[0]) < '2018':
                #print (item['sub_name'], str(date_vec[0]))
                #cnt += 1
            #print (item['sub_name'], date)
            if date in dict_xnat[spore_id]:
                print (item['sub_name'], date, '==')
                inxnat.append(1)
            else:
                inxnat.append(0)
                
    df['If on Xnat (no=0, yes=1)'] = inxnat
    df.to_csv(save_csv_path) 
    print (len(inxnat))
    print (sum(inxnat))
    print (cnt)
    
def get_exist_set(qa_csvs):
    '''This create the what we have in SPORE from QA report'''
    sess_list = []
    for tmp_csv in qa_csvs:
        #tmp_csv = qa_csvs[i]
        df = pd.read_csv(tmp_csv)
        for i,item in df.iterrows():
            if item['qa'] != item['qa']:
                sess_list.append(item['nifti_name'])
    return sess_list

def get_existcol(sess_list, xlsx_path, sheet_name, save_path):
    df = pd.read_excel(xlsx_path,  sheet_name = sheet_name)
    exist_list = []
    for i, item in df.iterrows():
        #print (item['studydate'], str(item['studydate']))
        date_vec = str(item['studydate'])[:10].split('-')
        date = str(date_vec[0]) + str(date_vec[1]) + str(date_vec[2])
        print (item['SPORE_ID'][6:] + 'time' + date + '.nii.gz')
        if item['SPORE_ID'][6:] + 'time' + date + '.nii.gz' in sess_list:
            exist_list.append('yes')
        else:
            exist_list.append('no')
    df['exist'] = exist_list
    df.to_csv(save_path)
                
    
def get_dataset_info(csv_path):
    '''This function is to count the number of subjects and the number of longitudinal data'''
#    item_list = os.listdir(data_root)
#    subj_list = [re.split('[gt]', x)[1] for x in item_list] # x == reg27937769677time20110101.nii.gz, so re.split('[gt]', x)[1] = 27937769677
    df = pd.read_csv(csv_path)
    subj_list = df['subject'].tolist()
    subj_set = set(subj_list)
    l_cnt = 0
    for subj in subj_set:
        if subj_list.count(subj) > 1:
            l_cnt += 1
    
    print (l_cnt, len(subj_set))

def move_time_spread(time_root, spread_root):  # for spore now
    subj_list = os.listdir(time_root)
    for i in range(len(subj_list)):
        print (i, len(subj_list))
        sub_path = time_root + '/' + subj_list[i]
        sess_list = os.listdir(sub_path)
        for j in range(len(sess_list)):
            #print (sub_path + '/' + sess_list[j])
            item_paths = glob(os.path.join(sub_path + '/' + sess_list[j], '*.nii.gz'))
            if len(item_paths) == 0:
                print (sub_path + '/' + sess_list[j], ' is empty')
                continue
            item_path = item_paths[0]
            new_name = subj_list[i] + 'time' + sess_list[j]
            mkdir(spread_root + '/' + new_name)
            print ('mv ' + item_path + ' ' + spread_root + '/' + new_name + '/' + new_name + '.nii.gz')
            os.system('mv ' + item_path + ' ' + spread_root + '/' + new_name + '/' + new_name + '.nii.gz')
            
def makesure_exclusion(csv_path):
    df = pd.read_csv(csv_path)
    tr_set = set([])
    val_set = set([])
    tt_set = set([])
    for i, item in df.iterrows():
        if item['trainvalnew'] == 0:
            tt_set.add(item['subject'])
        if item['trainvalnew'] > 0 and item['trainvalnew'] < 4:
            tr_set.add(item['subject'])
        if item['trainvalnew'] == 4:
            val_set.add(item['subject'])
    print (len(tr_set), len(val_set), len(tt_set))
    print (tr_set & tt_set, tr_set & val_set, tt_set & val_set)


def latest_item_kaggle(ori_csv, new_csv):
    '''
    deal with longitudinal data. select the newest CT for computing.
    exp:latest_item_kaggle('/share3/gaor2/share5backup/data/kaggle/results/MCL_regcombine_kaggle_result.csv', '/share3/gaor2/share5backup/data/kaggle/results/MCL_regcombine_latestitem.csv')
    '''
    df = pd.read_csv(ori_csv)
    subj_max = {}
    for i, item in df.iterrows():
        subj = re.split('[gt]', item['id'])[1]   # item['id'] like reg40404567372time20120503
        if subj not in subj_max.keys():
            subj_max[subj] = item['id']
        else:
            subj_max[subj] = max(item['id'], subj_max[subj])
    drop_list = []        
    for i, item in df.iterrows():
        subj = re.split('[gt]', item['id'])[1] 
        if subj_max[subj] != item['id']:
            drop_list.append(i)
    new_df = df.drop(drop_list)

    new_df.to_csv(new_csv)

def split_kaggle_set(label_path, kaggle_latestitem_path, save_val, save_test):
    
    df_label = pd.read_csv(label_path)
    df_kaggle = pd.read_csv(kaggle_latestitem_path)
    val_list, test_list = [], []
    gt_item = {}
    for i, item in df_label.iterrows():
        if item['trainvalnew'] == 4:
            val_list.append(item['item'])
        if item['trainvalnew'] == 0:
            test_list.append(item['item'])
        gt_item[item['item']] = item['y'] 
#    print (val_list, test_list)
    val_cancer, val_gt, val_item = [], [], []
    test_cancer, test_gt, test_item = [], [], []
    for i, item in df_kaggle.iterrows():
        nii_item = item['id'] + '.nii.gz'
        if nii_item in val_list:
            val_cancer.append(item['cancer'])
            val_gt.append(gt_item[nii_item])
            val_item.append(item['id'])
        if nii_item in test_list:
            test_cancer.append(item['cancer'])
            test_gt.append(gt_item[nii_item])
            test_item.append(item['id'])
            
    df_val = pd.DataFrame()
    df_val['item'] = val_item
    df_val['cancer'] = val_cancer
    df_val['gt'] = val_gt
    
    df_test = pd.DataFrame()
    df_test['item'] = test_item
    df_test['cancer'] = test_cancer
    df_test['gt'] = test_gt
    
    df_val.to_csv(save_val)
    df_test.to_csv(save_test)

def split_kaggle_set2(label_path, save_val, save_test):
    '''
    This one is for Jiachen's file final_4fold_addkaggle.csv
    '''
    val_cancer, val_gt, val_item = [], [], []
    test_cancer, test_gt, test_item = [], [], []
    df_label = pd.read_csv(label_path)
    for i, item in df_label.iterrows():
        if item['trainvalnew'] == 4:
            val_cancer.append(item['from_baseline'])
            val_gt.append(item['y'])
            val_item.append(item['subject'])
        if item['trainvalnew'] == 0:
            test_cancer.append(item['from_baseline'])
            test_gt.append(item['y'])
            test_item.append(item['subject'])
    df_val = pd.DataFrame()
    df_val['item'] = val_item
    df_val['cancer'] = val_cancer
    df_val['gt'] = val_gt
    
    df_test = pd.DataFrame()
    df_test['item'] = test_item
    df_test['cancer'] = test_cancer
    df_test['gt'] = test_gt
    
    df_val.to_csv(save_val)
    df_test.to_csv(save_test)
    
def cnt_ct_MCL(csv_path):
    '''The csv file is generated by xnat, this one for MCL'''
    df = pd.read_csv(csv_path)
    subj_ct = {}
    for i, item in df.iterrows():
        subj_name = re.split('[-_]', item['Subject'])[0]
        if subj_name not in subj_ct.keys():
            subj_ct[subj_name] = 0
        subj_ct[subj_name] += item['CT Sessions']
    cnt_list = list(subj_ct.values())
    cnt_list = [int (x) for x in cnt_list if x == x]
    plt.hist(cnt_list, range = (0, 15))
    plt.xlabel('sessions per subject')
    plt.ylabel('number of subjects')
    plt.show()
    print (len(cnt_list), sum(cnt_list))
    
def cnt_ct_SPORE(csv_path):
    df = pd.read_csv(csv_path)
    ct_list = df['CT Sessions'].tolist()
    plt.hist(ct_list, range(0, 15))
    plt.xlabel('sessions per subject')
    plt.ylabel('number of subjects')
    plt.show()
    print (len(ct_list), sum(ct_list))
    
def get_label_SPORE():
    dfm = pd.read_excel('/share3/gaor2/share5backup/data/SPORE/clinical/label_batch1_deidentify.xlsx', sheet_name = 'Malignant Paths')
    mal_list = dfm['SPORE_ID'].tolist()
    dfb = pd.read_excel('/share3/gaor2/share5backup/data/SPORE/clinical/label_batch1_deidentify.xlsx', sheet_name = 'Benign Paths')
    ben_list = dfb['SPORE_ID'].tolist()
    df = pd.read_csv('/share3/gaor2/share5backup/data/SPORE/SPOREnorm/info.csv')
    label_list = []
    for i, item in df.iterrows():
        if 'SPORE_' + str('%08d' % item['subject']) in mal_list:
            label_list.append(1)
        elif 'SPORE_' + str('%08d' % item['subject']) in ben_list:
            label_list.append(0)
        else:
            label_list.append('')
    df['y'] = label_list
    df.to_csv('/share3/gaor2/share5backup/data/SPORE/SPOREnorm/label.csv')
    
    
def add_label_NLST0(npy_root, cancer_csv, nocancer_csv, save_csv):
    '''Get the label csv file, this is from npy_root style'''
    
    npy_list = os.listdir(npy_root)
    cancer_list = pd.read_csv(cancer_csv)['Demographics.pid'].tolist()
    cancer_list = [str(x) for x in cancer_list]
    id_list = [x[:-4] for x in npy_list]
    label_list = []
    for i in range(len(npy_list)):
        tmp_vec = re.split('[te.]', npy_list[i])
        subj_name = tmp_vec[0]
        
        if subj_name in cancer_list:
            label_list.append(1)
        else:
            label_list.append(0)
    trainval_list = ['validation'] * len(npy_list)
    data = pd.DataFrame()
    data['id'] = id_list
    data['cancer'] = label_list
    data['trainval'] = trainval_list
    data.to_csv(save_csv)
    
def add_label_NLST(ori_csv, cancer_csv, nocancer_csv, save_csv):
    '''Get the label csv file '''
    
    df = pd.read_csv(ori_csv)
    cancer_list = pd.read_csv(cancer_csv)['subject'].tolist()
    nocancer_list = pd.read_csv(nocancer_csv)['subject'].tolist()
    #cancer_list = [str(x) for x in cancer_list]
    #id_list = [x[:-4] for x in npy_list]
    label_list = []
    gt_reg = []
    for i, item in df.iterrows():
        if item['subject'] in cancer_list and item['item'] == 1:
            label_list.append(1)
#         elif item['subject'] in nocancer_list:
#             label_list.append(0)
        else:
            label_list.append(0)
        if item['subject'] in cancer_list:
            gt_reg.append(1)
        else:
            gt_reg.append(0)
    df['gt_reg'] = gt_reg
    df['gt'] = label_list
    df.to_csv(save_csv)    
      
def rm_null_file(data_root):
    data_list = os.listdir(data_root)
    for i in range(len(data_list)):
        if os.path.getsize(data_root + '/' + data_list[i]) == 0:
            print (data_root + '/' + data_list[i])
            #os.system('rm ' + data_root + '/' + data_list[i])
    
def cnt_num_NLST(npy_root):
    npy_list = os.listdir(npy_root)
    subj_path_dict = {}
    for i in range(len(npy_list)):
        subj_name = npy_list[i].split('t')[0]
        if subj_name not in subj_path_dict.keys():
            subj_path_dict[subj_name] = []
        subj_path_dict[subj_name].append(npy_list[i])
    three_list, two_list, more_list = [], [], []
    for key in subj_path_dict.keys():
        if len(subj_path_dict[key]) == 3:
            three_list.append(key)
        if len(subj_path_dict[key]) == 2:
            two_list.append(key)
        if len(subj_path_dict[key]) >= 2:
            more_list.append(key)
    #print (len(three_list), len(two_list), len(subj_path_dict.keys()) - len(three_list) - len(two_list))
    return more_list

def get_lastitem_kagglecsv(ori_csv):
    df = pd.read_csv(ori_csv)
    subj_sess = {}
    for i, item in df.iterrows():
        subj = re.split('[t]', item['id'])[0]
        if subj not in subj_sess.keys():
            subj_sess[subj] = []
        subj_sess[subj].append(item['id'])
    for key in subj_sess.keys():
        subj_sess[key] = sorted(subj_sess[key])
    
    lastitem = []
    for i, item in df.iterrows():
        subj = re.split('[t]', item['id'])[0]
        if item['id'] == subj_sess[subj][-1]:
            lastitem.append(1)
        else:
            lastitem.append(0)
    df['lastitem'] = lastitem
    df.to_csv(ori_csv)

def get_lastitem_csv(ori_csv):
    df = pd.read_csv(ori_csv)
    subj_sess = {}
    for i, item in df.iterrows():
        subj = item['subject']
        if subj not in subj_sess.keys():
            subj_sess[subj] = []
        subj_sess[subj].append(item['id'])
    for key in subj_sess.keys():
        subj_sess[key] = sorted(subj_sess[key])
    
    lastitem = []
    for i, item in df.iterrows():
        subj = item['subject']
        if item['id'] == subj_sess[subj][-1]:
            lastitem.append(1)
        else:
            lastitem.append(0)
    df['lastitem'] = lastitem
    df.to_csv(ori_csv)
    
    
def add_lastitem_NLST(ori_csv, kaggle_res, new_csv):
    df = pd.read_csv(kaggle_res)
    id_risk = {}
    for i, item in df.iterrows():
        id_risk[item['id']] = item['cancer']
    subj_risk = {}
    for i, item in df.iterrows():
        if item['id'][0:6] + 'time2001' in id_risk.keys():
            subj_risk[item['id'][0:6]] = id_risk[item['id'][0:6] + 'time2001']
        elif item['id'][0:6] + 'time2000' in id_risk.keys():
            subj_risk[item['id'][0:6]] = id_risk[item['id'][0:6] + 'time2000']
        else:
            subj_risk[item['id'][0:6]] = id_risk[item['id'][0:6] + 'time1999']
    
    print (len(subj_risk.keys()))
    df_ori = pd.read_csv(ori_csv)
    kaggle_risk = []
    for i, item in df_ori.iterrows():
        if str(int(item['pid'])) not in subj_risk.keys():
            kaggle_risk.append('')
        else:
            kaggle_risk.append(subj_risk[str(int(item['pid']))])
    df_ori['kaggle_risk'] = kaggle_risk
    df_ori.to_csv(new_csv)

def get_lastitem_csv0(kaggle_csv, krnn_csv, knew_csv):
    '''This function depend on the krnn_csv, only work with rnn method'''
    last_list = pd.read_csv(krnn_csv)['id'].tolist()
    df = pd.read_csv(kaggle_csv)
    cancer_list, gt_list, id_list = [], [], []
    for i, item in df.iterrows():
        if item['id'] in last_list:
            cancer_list.append(item['cancer'])
            gt_list.append(item['gt'])
            id_list.append(item['id'])
    data = pd.DataFrame()
    data['cancer'] = cancer_list
    data['gt'] = gt_list
    data['id'] = id_list
    data.to_csv(knew_csv)
        
def select_longitudinal_set(ori_root, label_csvs, save_root):
    data_list = os.listdir(ori_root)
    subj1_list = pd.read_csv(label_csvs[0])['Lookup MCL'].tolist()
    subj2_list = pd.read_csv(label_csvs[1])['MCL.id'].tolist()
    subj_list = subj1_list + subj2_list
    new_list = []
    for i in range(len(data_list)):
        if len(os.listdir(ori_root + '/' + data_list[i])) > 1 and data_list[i] in subj_list:
            new_list.append(data_list[i])
            if not os.path.exists(save_root + '/' + data_list[i]):
                os.mkdir(save_root + '/' + data_list[i])
            print ('cp -r ' + ori_root + '/' + data_list[i] + ' ' + save_root + '/' + data_list[i])
            os.system('cp -r ' + ori_root + '/' + data_list[i] + '/* ' + save_root + '/' + data_list[i])
    print (new_list, len(new_list))
    
def select_with_label_SPORE(ori_root, label_csv, save_root):
    ori_list = os.listdir(ori_root)
    data_list = ['SPORE_' + x for x in ori_list]
    subj_list = pd.read_csv(label_csv)['SPORE_ID'].tolist()
    
    print (len(subj_list), ' total subjects')
    #print (subj_list)
    #print (data_list)
    new_list = []
    for i in range(len(data_list)):
        if data_list[i] in subj_list:
            new_list.append(data_list[i])
            if not os.path.exists(save_root + '/' + data_list[i]):
                os.mkdir(save_root + '/' + data_list[i])
            print ('cp -r ' + ori_root + '/' + ori_list[i] + '/* ' + save_root + '/' + data_list[i])
            os.system('cp -r ' + ori_root + '/' + ori_list[i] + '/* ' + save_root + '/' + data_list[i])
    print (new_list, len(new_list))

    
def get_longitudinal_NLST(npy_root, label_csv):
    '''This function split the NLST with train/val/test and generate the long_NLST.csv'''
    #get_longitudinal_NLST('/share3/gaor2/share5backup/data/NLST/featnpy', '/share3/gaor2/share5backup/data/NLST/fortest_rnnNLST.csv')
#delete_row('/share3/gaor2/share5backup/data/NLST/fortest_rnnNLST.csv', '/share3/gaor2/share5backup/data/NLST/long_NLST.csv')
    long_list = cnt_num_NLST(npy_root)
    #long_list = three_list + two_list
    trainval = {'train': [], 'val': [], 'test': []}
    for i in range(len(long_list)):
        if i % 18 < 12:
            trainval['train'].append(long_list[i])
        if i % 18 in [12, 13, 14, 15]:
            trainval['test'].append(long_list[i])
        if i % 18 in [17, 16]:
            trainval['val'].append(long_list[i])
    df = pd.read_csv(label_csv)
    phase_list = []
    for i, item in df.iterrows():
        subj_name = item['id'].split('t')[0]
        if subj_name in trainval['train']:
            phase_list.append('train')
        elif subj_name in trainval['val']:
            phase_list.append('val')
        elif subj_name in trainval['test']:
            phase_list.append('test')
        else:
            phase_list.append('')
    df['phase'] = phase_list
    df.to_csv(label_csv)
    

    
def get_long_NLST(label_csv, new_label):
    '''This function split the NLST with train/val/test and generate the long_NLST.csv, very similar to get_longitudinal_NLST except the input file'''
    df = pd.read_csv(label_csv)
    subj_sess_dict = {}

    for i, item in df.iterrows():
        if item['subject'] not in subj_sess_dict.keys():
            subj_sess_dict[item['subject']] = []
        subj_sess_dict[item['subject']].append(item['session'])

    long_list = []    

    for key in subj_sess_dict.keys():
        if len(subj_sess_dict[key]) >= 2:
            long_list.append(key)

    trainval = {'train': [], 'val': [], 'test': []}
    for i in range(len(long_list)):
        if i % 18 < 16:
            trainval['train'].append(long_list[i])
#         if i % 18 in [ 13, 14, 15]:
#             trainval['test'].append(long_list[i])
        if i % 18 in [17, 16]:
            trainval['val'].append(long_list[i])
    print (len(trainval['train']), len(trainval['val']),  len(trainval['test']))

    df = pd.read_csv(label_csv)
    phase_list = []
    for i, item in df.iterrows():
        subj_name = item['subject']
        if subj_name in trainval['train']:
            phase_list.append('train')
        elif subj_name in trainval['val']:
            phase_list.append('val')
        elif subj_name in trainval['test']:
            phase_list.append('test')
        else:
            phase_list.append('')
    df['new_phase'] = phase_list
    df.to_csv(new_label)

def split_NLST_set(ori_csv, subset_csv, phase):
    df = pd.read_csv(ori_csv)
    new_id, new_gt, new_pred = [], [], []
    
    for i, item in df.iterrows():
        if item['phase'] == phase:
            new_id.append(item['id'])
            new_gt.append(item['gt'])
            new_pred.append(item['pred'])
    data = pd.DataFrame()
    data['id'] = new_id
    data['gt'] = new_gt
    data['pred'] = new_pred
    data.to_csv(subset_csv)
  
def get_feats_h5py(data_root, txt_path, save_path):
    f = open(txt_path, 'r')
    lines = f.readlines()
    paths = [line.split(' ')[0] for line in lines]
    labels = [int(line.strip().split(' ')[1]) for line in lines]
    #np.random.shuffle(lines)
    print (len(paths))
    data = np.random.rand(len(paths), 256).astype('float32')
    for i in range(len(lines)):
        if i % 10000 == 0: print ('line: ', i, '/', len(paths))
        #print (data_root + '/' + lines[i])
        data[i] = np.fromfile(data_root + '/' + paths[i], dtype = np.float32)
    f = h5py.File(save_path, 'w')
    f.create_dataset('feats', data = data)
    f.create_dataset('paths', data = np.string_(paths))
    f.create_dataset('labels', data = labels)
    #f.create_dataset('paths', data = lines)
    f.close()
            
def get_imglist_h5py(txt_path, save_path):
    f = open(txt_path, 'r')
    lines = f.readlines()
    paths = [line.split(' ')[0] for line in lines]
    labels = [int(line.strip().split(' ')[1]) for line in lines]
    #np.random.shuffle(lines)
    print (len(lines))
#     data = np.random.rand(len(lines), 256).astype('float32')
#     for i in range(len(lines)):
#         if i % 100 == 0: print (i)
#         #print (data_root + '/' + lines[i])
#         data[i] = np.fromfile(data_root + '/' + lines[i], dtype = np.float32)
    df = h5py.File(save_path, 'w')
    df.create_dataset('paths', data = np.string_(paths))
    df.create_dataset('labels', data = labels)
    #f['paths'] = paths
    #f['labels'] = labels
    df.close()
    
def add_timepoint_NLST(ori_csv, long_csv):
    df = pd.read_csv(ori_csv)
    pid_time_dict = {}
    for i, item in df.iterrows():
        pid_time_dict[item['pid']] = [item['scr_days0'], item['scr_days1'],item['scr_days2']]
    df_long = pd.read_csv(long_csv)
    time_list = []
    for i, item in df_long.iterrows():
        tmp_list = pid_time_dict[int(item['id'][:6])]
        tmp_max = max(tmp_list)#tmp_list[1] if tmp_list[2] != tmp_list[2] else tmp_list[2]
        if item['id'][-4:] == '1999':
            tmp_time = tmp_max - tmp_list[0]
        if item['id'][-4:] == '2000':
            tmp_time = tmp_max - tmp_list[1]
        if item['id'][-4:] == '2001':
            tmp_time = tmp_max - tmp_list[2]
        tmp_time = tmp_time / 365.
        time_list.append(tmp_time)
    df_long['time_dis'] = time_list
    df_long.to_csv(long_csv, index = False)  
    
def add_time_diag_NLST(ori_csv, long_csv):
    df = pd.read_csv(ori_csv)
    diag_dict = {}
    for i, item in df.iterrows():
        diag_dict[item['pid']] = [item['scr_days0'], item['scr_days1'],item['scr_days2'], item['fup_days'], item['candx_days']]
    df_long = pd.read_csv(long_csv)
    time_list, diag_list, fup_days, candx_days  = [], [], [], []
    for i, item in df_long.iterrows():
        tmp_list = diag_dict[int(item['subject'])]
#         if item['sess'] not in [1999, 2000, 2001]:
#             print (item['id'], item['sess'])
        if item['sess'] == '1999':
            tmp_time = tmp_list[0]
        if item['sess'] == '2000':
            tmp_time = tmp_list[1]
        if item['sess'] == '2001':
            tmp_time =  tmp_list[2]

        fup_days.append(tmp_list[3])

        candx_days.append(tmp_list[4])
        time_list.append(tmp_time)

    df_long['time'] = time_list
    df_long['fup_days'] = fup_days
    df_long['candx_days'] = candx_days
    df_long.to_csv(long_csv, index = False)
    
def save_nii_asnpy(ori_root, new_root):
    data_list = os.listdir(ori_root)
    for i in range(len(data_list)):
        if i % 50 == 0: print (i)
        data_3d = nib.load(ori_root + '/' + data_list[i])
        data = data_3d.get_data()
        np.save(new_root + '/' + data_list[i].replace('.nii.gz', '.npy'), data)
        

def move_labeled_data(ori_root, new_root, label_list):
    ori_list = os.listdir(ori_root)
    for i in range(len(label_list)):
        if label_list[i] in ori_list:
            print (i, len(label_list), 'mv ' + ori_root + '/' + label_list[i] + ' ' + new_root)
            os.system('mv ' + ori_root + '/' + label_list[i] + ' ' + new_root)
        

    
def get_info_img(img_path, save_csv, source):
    data_list = os.listdir(img_path)
    source_list, item_list, subj_list, sess_list = [], [], [], []
    for i in range(len(data_list)):
        source_list.append(source)
        item_list.append(data_list[i])
        tmp_id = data_list[i].replace('.nii.gz', '')
        tmp_vec = re.split('[time]', tmp_id)
        subj_list.append(tmp_vec[0])
        sess_list.append(tmp_vec[-1])
    data = pd.DataFrame()
    data['item'] = item_list
    data['source'] = source_list
    data['subject'] = subj_list
    data['sess'] = sess_list
    data.to_csv(save_csv, index = False)

    
# 4 steps in the following to get the label.csv file for train, this is for threeset  
    
    
def get_info_feat(npy_path, save_csv):
    data_list = os.listdir(npy_path)
    source_list, id_list, subj_list, sess_list = [], [], [], []
    
    for i in range(len(data_list)):
        tmp_id = data_list[i][:-4]
        tmp_vec = re.split('[time]', tmp_id)
        if len(tmp_vec[0]) == 6:
            source_list.append('nlst')
        elif tmp_vec[0][:2] == '00':
            source_list.append('spore')
        else:
            source_list.append('mcl')
        
        subj_list.append(tmp_vec[0])
        sess_list.append(tmp_vec[-1])
        
        id_list.append(tmp_id)
        
    data = pd.DataFrame()
    data['id'] = id_list
    data['source'] = source_list
    data['subject'] = subj_list
    data['sess'] = sess_list
    data.to_csv(save_csv, index = False)

def get_mcl_label(MCL_csvs, MCL_xlsx, save_csv):
    id_label = {'0': [], '1': [], '-1': []}
    MCLdf0 = pd.read_csv(MCL_csvs[1])
    for i, item in MCLdf0.iterrows():
        if item['Histologic Type'] != item['Histologic Type']:
            continue
        if item['Histologic Type'].strip() in ['Adenocarcinoma', 'Large Cell Neuroendocrine', 'Non Small Cell (NSCLC)', 'Small Cell Carcinoma', 'Squamous Cell Carcinoma']:
            id_label['1'].append(str(item['Lookup MCL']))
        if item['Histologic Type'].strip() in ['Adenoid Cystic Carcinoma', 'Adenosquamous Carcinoma', 'Atypical Carcinoid', 'Carcinoid', 'Stage IB', 'Stage IIB', 'other', 'Other', 'IGNORE', 'No Diagnosis']:
            id_label['-1'].append(str(item['Lookup MCL']))
        if item['Histologic Type'].strip() in ['Granuloma', 'Negative for Dysplasia and Metaplasia', 'Negative for Malignant Cells', 'Squamous Metaplasia', 'Normal']:
            id_label['0'].append(str(item['Lookup MCL']))
    
    MCLdf1 = pd.read_csv(MCL_csvs[0])
    for i, item in MCLdf1.iterrows():
        if item['Histologic.Type'] != item['Histologic.Type']:
            continue
        if item['Histologic.Type'].strip() in ['Adenocarcinoma', 'Large Cell Neuroendocrine', 'Non Small Cell (NSCLC)', 'Small Cell Carcinoma', 'Squamous Cell Carcinoma']:
            id_label['1'].append(str(item['MCL.id']))
        if item['Histologic.Type'].strip() in ['Adenoid Cystic Carcinoma', 'Adenosquamous Carcinoma', 'Atypical Carcinoid', 'Carcinoid', 'Stage IB', 'Stage IIB', 'other', 'Other', 'IGNORE', 'No Diagnosis']:
            id_label['-1'].append(str(item['MCL.id']))
        if item['Histologic.Type'].strip() in ['Granuloma', 'Negative for Dysplasia and Metaplasia', 'Negative for Malignant Cells', 'Squamous Metaplasia', 'Normal' ]:
            id_label['0'].append((item['MCL.id']))
        
    df = pd.read_excel(MCL_xlsx, sheet_name = 'update.all.dat1')   
    for i, item in df.iterrows():
        if item['Histologic.Type'] != item['Histologic.Type']:
            continue
        if item['Histologic.Type'].strip() in ['Adenocarcinoma', 'Large Cell Neuroendocrine', 'Non Small Cell (NSCLC)', 'Small Cell Carcinoma', 'Squamous Cell Carcinoma']:
            id_label['1'].append(str(item['Image.ID']))
        if item['Histologic.Type'].strip() in ['Adenoid Cystic Carcinoma', 'Adenosquamous Carcinoma', 'Atypical Carcinoid', 'Carcinoid', 'Stage IB', 'Stage IIB', 'other', 'Other', 'IGNORE', 'No Diagnosis']:
            id_label['-1'].append(str(item['Image.ID']))
        if item['Histologic.Type'].strip() in ['Granuloma', 'Negative for Dysplasia and Metaplasia', 'Negative for Malignant Cells', 'Squamous Metaplasia', 'Normal' ]:
            id_label['0'].append((item['Image.ID']))
            
    pos_list = list(set(id_label['1']))

    neg_list = list(set(id_label['0']) - set(id_label['1']))

    data = pd.DataFrame()

    id_list = pos_list + neg_list

    label_list = [1] * len(pos_list) + [0] * len(neg_list)

    data['id'] = id_list
    data['gt'] = label_list

    data.to_csv(save_csv, index = False)
    
def add_label_MCL(info_csv, mcl_label):
    df = pd.read_csv(mcl_label)
    withlabel_list = df['id'].tolist()
    subj_dict = {0: [], 1: []}
    for i, item in df.iterrows():
        subj_dict[item['gt']].append(item['id'])
    df_data = pd.read_csv(info_csv)
    label_list = []
    #print (withlabel_list)
    for i,item in df_data.iterrows():
        if str(item['subject']) not in withlabel_list:
            label_list.append('')
        elif str(item['subject']) in subj_dict[1] and item['lastitem'] == 1:
            label_list.append(1)
        elif str(item['subject']) in subj_dict[0] or item['lastitem'] == 0:
            label_list.append(0)
    df_data['gt'] = label_list
    df_data.to_csv(info_csv, index = False)
            
    
    
def add_label_MCLSpore(info_csv, mcl_label0, mcl_label, spore_label, nlst_cancer, nlst_nocancer, save_label):
    # for featnpy to get label
    # this for mcl label is wrong, please refer to get_mcl_label and id_label_dict. ## 20190522. The threeset label need small update.
    subj_label_dict = {}
    
    df_mcl0 = pd.read_csv(mcl_label0)
    for i,item in df_mcl0.iterrows():
        if item['Histologic Type'] in ['Normal', 'Negative for Dysplasia and Metaplasia', 
                                  'Negative for Malignant Cells','Granuloma ', 'Granuloma','Squamous Metaplasia']:
            subj_label_dict[str(item['Lookup MCL'])] = 0
        else:
            subj_label_dict[str(item['Lookup MCL'])] = 1

    df_mcl = pd.read_csv(mcl_label)
    for i,item in df_mcl.iterrows():
        if item['Histologic.Type'] in ['Normal', 'Negative for Dysplasia and Metaplasia', 
                                  'Negative for Malignant Cells','Granuloma ', 'Granuloma','Squamous Metaplasia']:
            subj_label_dict[str(item['MCL.id'])] = 0
        else:
            subj_label_dict[str(item['MCL.id'])] = 1
            
    
            
    #df_spore = pd.read_csv(spore_label) #'/share3/gaor2/share5backup/data/SPORE/clinical/label_batch1_deidentify.xlsx'
    dfm = pd.read_excel(spore_label, sheet_name = 'Malignant Paths')
    mal_list = dfm['SPORE_ID'].tolist()
    for sporeid in mal_list:
        subj_label_dict[str(sporeid)] = 1
        
    
    cancer_list = pd.read_csv(nlst_cancer)['Demographics.pid'].tolist()
    nocancer_list = pd.read_csv(nlst_nocancer)['Demographics.pid'].tolist()
    for subj in cancer_list:
        subj_label_dict[str(subj)] = 1
    for subj in nocancer_list:
        subj_label_dict[str(subj)] = 0
     
        
    df_info = pd.read_csv(info_csv)
    
    label_list = []
    for i, item in df_info.iterrows():
        subject = item['item'].split('t')[0]
        if item['source'] == 'spore':
            sporeid = 'SPORE_' + '%08d' % int(subject)
            if sporeid in subj_label_dict.keys():
                print (sporeid)
                label_list.append(1)
            else:
                label_list.append(0)
            
        else:
            if str(subject) in subj_label_dict:
                label_list.append(subj_label_dict[str(subject)])
            else:
                label_list.append('')
    df_info['gt'] = label_list            
    df_info.to_csv(save_label)
    
def get_three_phase(label_csv, new_label):
    '''This function split the threeset verion'''
    df = pd.read_csv(label_csv)
    
    phase_dict = {}
    
    df_proj = df.loc[df['source'] != 'nlst']
    proj_list = list(set(df_proj['subject'].tolist()))
    for i in range(len(proj_list)):
        phase_dict[proj_list[i]] = i % 5
    
    df_nlst = df.loc[df['source'] == 'nlst']
    nlst_list = list(set(df_nlst['subject'].tolist()))
    for i in range(len(nlst_list)):
        phase_dict[nlst_list[i]] = i % 5
    

    phase_list = []
    for i, item in df.iterrows():
        phase_list.append(phase_dict[item['subject']])
    df['phase'] = phase_list
    df.to_csv(new_label, index = False)

from datetime import date    
    
def get_three_time(nlst_csv, label_csv):
    df_nlst = pd.read_csv(nlst_csv)
    id_time_dict = {}
    for i, item in df_nlst.iterrows():
        
        id_time_dict[item['item']] = item['norm_time']
                  
    
    df = pd.read_csv(label_csv)
    subj_sess_dict = {}
    for i,item in df.iterrows():
        if item['source'] != 'nlst':
            subj = item['subject']
            if subj not in subj_sess_dict.keys():
                subj_sess_dict[subj] = []
            subj_sess_dict[subj].append(item['sess'])
        
    time_list = []    
    for i, item in df.iterrows():
        if item['source'] == 'nlst':
            if item['item'] not in id_time_dict.keys():
                print (i, item['item'])
                time_list.append('')
            else:
                time_list.append(id_time_dict[item['item']])
        else:
            base_datestr = str(min(subj_sess_dict[item['subject']]))
            sess_datestr = str(item['sess'])
            #print (base_datestr, sess_datestr)
            base_date = date(int(base_datestr[:4]), int(base_datestr[4:6]), int(base_datestr[6:8]))
            sess_date = date(int(sess_datestr[:4]), int(sess_datestr[4:6]), int(sess_datestr[6:8]))
            delta = sess_date - base_date
            time_list.append(delta.days / 365.)
    df['norm_time'] = time_list
    df.to_csv(label_csv, index = False)
    
def convert_diag_to_mclcsv():
    from datetime import date 
    df = pd.read_csv('/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/data/CoTrain/mcl_label.csv')
    #diag = df['diag'].tolist()
    diag_dis = []
    for i, item in df.iterrows():
        if item['diag'] != item['diag']:
            diag_dis.append('')
        else:
            diag_vec = str(item['diag']).split('/')
            diag_date = date(int(diag_vec[-1]), int(diag_vec[0]), int(diag_vec[1]))
            sess_date = date(int(str(item['sess'])[:4]), int(str(item['sess'])[4:6]), int(str(item['sess'])[6:8]))
            delta = diag_date - sess_date
            diag_dis.append(delta.days / 365.0)

    df['diag_dis'] = diag_dis
    df.to_csv('/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/data/CoTrain/mcl_label.csv', index = False)
    
def combine_kaggle_pred():
    sess_prob_dict = {}

    data_list = os.listdir('/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/data/DSB_File/prep')

    df_nlst = pd.read_csv('/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/data/NLST/long_NLST.csv')

    for i,item in df_nlst.iterrows():
        sess_prob_dict[item['id']] = item['kaggle_cancer']

    df_mcl = pd.read_csv('/share2/gaor2/MCL/kaggle_result_noempty_old.csv')

    for i,item in df_mcl.iterrows():
        sess_prob_dict[item['id']] = item['cancer']

    df_spore = pd.read_csv('/share2/gaor2/SPORE/kaggle_result.csv')

    for i,item in df_spore.iterrows():
        sess_prob_dict[item['id']] = item['cancer']

    sess_prob_second = {}  # second is the second consideration

    df_second = pd.read_csv('/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/data/TwoLungSet/kaggle_result_noempty.csv')

    for i,item in df_second.iterrows():
        sess_prob_second[item['id']] = item['cancer']

    df_second_2 = pd.read_csv('/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/data/TwoLungSet/second/kaggle_result.csv')

    for i,item in df_second_2.iterrows():
        sess_prob_second[item['id']] = item['cancer']

    sess_prob_third = {}  # second is the last consideration

    df_third = pd.read_csv('/share5/gaor2/data/MCL/MCL_kaggle_reg.csv')

    for i,item in df_third.iterrows():
        sess_prob_third[item['id'][3:]] = item['cancer']

    df = pd.read_csv('/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/data/ThreeSet/label_noempty.csv')
    kaggle_pred = []
    cnt = 0
    for i, item in df.iterrows():
        if item['id'] in sess_prob_dict.keys():
            prob = sess_prob_dict[item['id']]
        elif item['id'] in sess_prob_second.keys():
            prob = sess_prob_second[item['id']]
            #print (item['id'])
        elif item['id'] in sess_prob_third.keys():
            prob = sess_prob_third[item['id']]
        else:
            prob = 0.5
        kaggle_pred.append(prob)
    df['cancer'] = kaggle_pred
    df.to_csv('/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/data/ThreeSet/label_noempty.csv', index = False)
    print (cnt)
#print (sess_prob_third.keys())
    
def get_noexisted_img():
    img_list = os.listdir('/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/data/NLST/DSB_File/noreg/prep')

    df = pd.read_csv('/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/data/NLST/NIFTI_cancer.csv')

    noreg_norm = []

    for i, item in df.iterrows():
        if str(item['subject']) + 'time' + str(item['session']) + '_clean.npy' in img_list:
            noreg_norm.append(1)
        else:
            noreg_norm.append(0)

    df['noreg_norm'] = noreg_norm
    print (sum(noreg_norm), len(noreg_norm))
    df.to_csv('/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/data/NLST/NIFTI_cancer.csv', index = False)
#-----------------------------------------------------------------------    
    from glob import glob
    df_noexist = df.loc[df['noreg_norm'] == 0]

    for i, item in df_noexist.iterrows():
        ori_path = glob('/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/data/NLST/NIFTI_cancer/' \
                    + str(item['subject']) + '/*' + str(item['session']) + '*/*.nii.gz' )[0]
        Id = str(item['subject']) + 'time' + str(item['session'])
        os.mkdir('/share3/gaor2/share5backup/data/NLST/NLST_second/cancer_spread/' + Id)
        os.system('cp '+ ori_path + ' ' + '/share3/gaor2/share5backup/data/NLST/NLST_second/cancer_spread/' + Id + '/' + Id + '.nii.gz')
    
    
def get_new_label(csv_path):
    # this is the way that after discuss with dr. Landman, only the last scan can be the cancer scan. 
    df = pd.read_csv(csv_path)
    new_gt = []

    for i, item in df.iterrows():
        if item['lastitem'] == 0:
            new_gt.append(0)
        else:
            new_gt.append(item['gt'])

    df['new_gt'] = new_gt
    df.to_csv(csv_path, index = False)
    
def get_new_diag(csv_path):
    df = pd.read_csv(csv_path)
    new_diag = []
    
    for i, item in df.iterrows():
        if item['gt'] == 0:                # actually this one is the old gt. consistent for each subject
            new_diag.append(20.0)
        else:
            new_diag.append(item['diag_dis'])
        
    df['new_diag_dis'] = new_diag
    df.to_csv(csv_path, index = False)
    
def copy_update_file(ori_root, update_txt, new_root):
    f = open(update_txt)
    lines = f.readlines()
    need_list = [line.strip() for line in lines]
    subj_list = os.listdir(ori_root)
    for i in range(104, len(subj_list)):
        subj = subj_list[i]
        print (i, len(subj_list), subj)
        subj_path = ori_root + '/' + subj
        sess_list = os.listdir(subj_path)
        for sess in sess_list:
            if sess not in need_list:
                continue    
            sess_path = subj_path + '/' + sess
            nifti_root = sess_path + '/new_max/new_NIFTI'
            nifti_paths = glob(nifti_root + '/*.nii.gz')
            assert len(nifti_paths) == 1
            nifti_path = nifti_paths[0]
            os.rename(nifti_path, nifti_root + '/max.nii.gz')
            date = re.split('[-_]', sess)[-1]
            mkdir(new_root + '/' + subj + '/' + date)
            new_name = new_root + '/' + subj + '/' + date + '/' + subj + 'time' + date + '.nii.gz'
            print ('cp '+ nifti_root + '/max.nii.gz' + ' ' + new_name)
            os.system('cp '+ nifti_root + '/max.nii.gz' + ' ' + new_name)
            #break
        #break
            #
    
def correct_prep(data_root, new_root):
    data_list = os.listdir(data_root)
    for i in range(0, len(data_list)):
        print (i, data_list[i])
#         if (i > 936): 
#             print ('i is 936, breaked')
#             break
        clean_npy = np.load(data_root + '/' + data_list[i])[0]
        if data_list[i][-10:] == '_clean.npy':
            print (clean_npy.shape)
            clean_npy = np.rot90(clean_npy, -1, axes = (1,2))
            clean_npy = np.swapaxes(clean_npy, 1,2)
            clean_npy = np.swapaxes(clean_npy, 0, 2)
            clean_npy = np.expand_dims(clean_npy, axis = 0)
            np.save(new_root + '/' + data_list[i], clean_npy)
        else:
            print (clean_npy)
            #break
        #print (clean_npy.shape)
        #break

def update_mcl_label():
    '''
    Since in previous, some diagnosis have a little problem, this function can update the mcl label. But have not put in use yet. 06102019
    '''
    
    df = pd.read_csv('/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/data/CoTrain/threeset_bldiag.csv')
    df_label = pd.read_csv('/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/data/MCL/csv/clinical/new_mcl_label.csv')
    pos_list = df_label.loc[df_label['gt'] == 1]['id'].tolist()
    neg_list = df_label.loc[df_label['gt'] == 0]['id'].tolist()

    new_gt = []
    for i, item in df.iterrows():
        if item['source'] != 'mcl':
            new_gt.append(item['gt'])
        else:
            if str(item['subject']) in pos_list:
                new_gt.append(1)
            elif str(item['subject']) in neg_list:
                new_gt.append(0)
            else:
                new_gt.append('')
                print (str(item['subject']))
    df['new_gt'] = new_gt
    df.to_csv('/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/data/CoTrain/threeset_bldiag.csv', index = False)
    
def mv_needed_session():
    '''
    This function gives a txt file with needed session, and copy those to a new folder. 
    '''
    import os
    import re

    f = open('/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/data/MCL/SLICEdir/bad_slicedir_mclnorm_061119.txt')
    new_root = '/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/data/MCL/xnat/Xnat061119'
    lines = f.readlines()

    lines = [line.strip() for line in lines]

    data_root = '/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/data/MCL/xnat'

    data_list = os.listdir(data_root)
    cnt = 0
    for date in data_list:
        if date == 'Xnat061119':
            continue
        date_root = data_root + '/' + date + '/MCL'
        subj_list = os.listdir(date_root)
        for subj in subj_list:
            subj_path = date_root + '/' + subj
            sess_list = os.listdir(subj_path)
            for sess in sess_list:
                sess_vec = re.split('[-_]', sess)
                new_sess = sess_vec[0] + 'time' + sess_vec[-1]
                if new_sess in lines:
                    print (new_sess)
                    cnt += 1
                    print (cnt)
                    os.system('cp -r ' + subj_path + '/' + sess + ' ' + new_root)
                    
def diag2label_mcl(label_csv, clc_csv, clc_xlsx):
    '''
    This function add the diag information to the ori label csv file
    '''
    diag_dict = {}
    df0 = pd.read_csv(clc_csv)
    for i, item in df0.iterrows():
        if len(str(item['Diagnosis.Date'])) > 2:
            diag_dict[str(item['MCL.id'])] = item['Diagnosis.Date']
        
    df1 = pd.read_excel(clc_xlsx, sheet_name = 'update.all.dat1') 
    for i, item in df1.iterrows():
        if len(str(item['Diagnosis.Date.1'])) > 2:
            diag_dict[str(item['MCL.ID'])] = item['Diagnosis.Date.1']
    
    diag_list = []
    df = pd.read_csv(label_csv)
    for i, item in df.iterrows():
        if str(item['id']) not in diag_dict.keys():
            diag_list.append('')
        else:
            diag_list.append(diag_dict[str(item['id'])])
    df['diag_date'] = diag_list
    df.to_csv(label_csv,index = False)

def add_diag_mcl():
    '''
    This function add the diag info to training label file
    '''
    df = pd.read_csv('/share2/gaor2/CoTrain/new_mcl_label.csv')
    subj_date = {}
    for i, item in df.iterrows():
        subj_date[str(item['id'])] = item['diag_date']

    df_l = pd.read_csv('/share2/gaor2/CoTrain/mcl_label.csv')
    diag_date = []
    for i, item in df_l.iterrows():
        diag_date.append(subj_date[str(item['subject'])])
    df_l['diag_date'] = diag_date
    df_l.to_csv('/share2/gaor2/CoTrain/mcl_label.csv', index = False)

from datetime import date    
    
def get_diag_distance(csv_path):
    def get_diag_date(in_str):
        if in_str != in_str or len(in_str) == 0:
            return None
        in_vec = re.split('[/]', in_str.strip())
        if len(in_vec) == 3:
            diag_date = date(int(in_vec[2]), int(in_vec[0]), int(in_vec[1]))
        else:
            in_vec = re.split('[- ]', in_str)
            if len(in_vec) > 3:
                diag_date = date(int(in_vec[0]), int(in_vec[1]), int(in_vec[2]))
            else:
                diag_date = None
        return diag_date
            
    df = pd.read_csv(csv_path)
    diag_dis = []
    
    for i, item in df.iterrows():
        #print (item['diag_date'])
        diag_date = get_diag_date(item['diag_date'])
        if diag_date == None:
            diag_dis.append(1)
        else:
            sess_str = str(item['sess'])
            sess_date = date(int(sess_str[:4]), int(sess_str[4:6]), int(sess_str[6:8]))
            delta = diag_date - sess_date
            if delta.days / 365.0 < -10:
                print (item['subject'])
            diag_dis.append(delta.days / 365.0)
        
    df['diag_dis'] = diag_dis
    df.to_csv(csv_path, index = False)


def add_islong(csv_path):
    subj_sess = {}
    df = pd.read_csv(csv_path)
    for i, item in df.iterrows():
        if item['subject'] not in subj_sess.keys():
            subj_sess[item['subject']] = []
        subj_sess[item['subject']].append(item['sess'])
    islong_list = []
    for i, item in df.iterrows():
        if len(subj_sess[item['subject']]) <= 1:
            islong_list.append(0)
        else:
            islong_list.append(1)
    df['islong'] = islong_list
    df.to_csv(csv_path, index = False)
    
def add_islong_res():
    df = pd.read_csv('/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/data/ThreeSet/label_noempty.csv')
    islong_list = df.loc[df['islong'] == 1]['id'].tolist()
    csv_path='/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/saved_file/ThreeSet/v4/new_nlst/extenal/drnn/nlstval4_DisCRNNClassifier.csv'
    df1 = pd.read_csv(csv_path)

    islong = []

    for i, item in df1.iterrows():
        if item['img'] in islong_list:
            islong.append(1)
        else:
            islong.append(0)
    df1['islong'] = islong
    df1.to_csv(csv_path, index = False)
    
    
def add_long_lastitem_result(csv_path):
    subj_sess = {}
    df = pd.read_csv(csv_path)
    for i, item in df.iterrows():
        item_vec = re.split('[time]', item['id'])
        subj, sess = item_vec[0], item_vec[-1]
        if subj not in subj_sess.keys():
            subj_sess[subj] = []
        subj_sess[subj].append(sess)
    print (subj_sess)
    lastitem, islong = [], []
    for i, item in df.iterrows():
        item_vec = re.split('[time]', item['id'])
        subj, sess = item_vec[0], item_vec[-1]
        if sess == max(subj_sess[subj]):
            lastitem.append(1)
        else:
            lastitem.append(0)
        if len(subj_sess[subj]) > 1:
            islong.append(1)
        else:
            islong.append(0)
    df['islong'] = islong
    df['lastitem'] = lastitem
    df.to_csv(csv_path, index = False)
        
def gt_tmp_new():
    df = pd.read_csv('/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/data/NLST/NLSTnorm/noreg/all_labels.csv')
    id_gt, id_lastitem = {}, {}

    for i, item in df.iterrows():
        id_gt[item['item'].replace('.nii.gz', '')] = item['gt']
        id_lastitem[item['item'].replace('.nii.gz', '')] = item['lastitem']

    df1= pd.read_csv('/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/data/NLST/DSB_File/noreg/tmp_new.csv')

    gt_list, lastitem_list = [], []
    for i, item in df1.iterrows():
        gt_list.append(id_gt[item['id']])
        lastitem_list.append(id_lastitem[item['id']])

    df1['gt'] = gt_list
    df1['lastitem'] = lastitem_list
    df1.to_csv('/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/data/NLST/DSB_File/noreg/tmp_new.csv', index = False)

def combine_two_csv():
    df0 = pd.read_csv('/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/data/ThreeSet/label_include_auc0.99.csv')
    df1 = pd.read_csv('/media/gaor2/8e7f6ccf-3585-4815-856e-80ce8754c5b5/data/ThreeSet/nlst_update_include.csv')
    df0 = df0.loc[df0['source'] != 'nlst']
    data = pd.DataFrame()
    for item in ['id', 'source', 'subject', 'sess', 'phase', 'norm_time']:
        tmp_list = df0[item].tolist() + df1[item].tolist()
        data[item] = tmp_list
    data.to_csv(csv_path, index = False)
    
def update_blreg_csv1():
    df = pd.read_csv('/share2/gaor2/CoTrain/threeset_bldiag_new.csv')
    subj_time = {}
    pos_subj = []
    for i, item in df.iterrows():
        if item['subject'] not in subj_time.keys():
            subj_time[item['subject']] = []
        subj_time[item['subject']].append(item['norm_time'])
        if item['gt'] == 1:
            pos_subj.append(item['subject'])

    time_dis, new_diag = [], []
    for i, item in df.iterrows():
        time_dis.append(max(subj_time[item['subject']]) - item['norm_time'])
        if item['subject'] in pos_subj:
            new_diag.append(item['diag_dis'])
        else:
            new_diag.append(max(subj_time[item['subject']]) - item['norm_time'] + 1)

    df['time_dis'] = time_dis
    df['new_diag'] = new_diag
    df.to_csv('/share2/gaor2/CoTrain/threeset_bldiag_new.csv', index = False)
    
def get_2Dnorm(pbb_path, img_path, num_patch, img_save, mask_save, sample_size = 256):
    img = nib.load(img_path).get_data()
    mask = nib.load(img_path.replace('img', 'mask')).get_data()
    pbb = np.load(pbb_path)
    pbb = pbb[pbb[:, 0] > -1]
    pbb = nms(pbb, 0.05)
    boxes = pbb[:num_patch]
    print (img.shape[1:])
    img_2D = np.zeros((3 * num_patch, sample_size, sample_size))
    mask_2D = np.zeros((3 * num_patch, sample_size, sample_size))
    for i in range(len(boxes)):
        box = boxes[i].astype('int')[1:]
        try:
            tmp_img_a = img[box[0]]
            tmp_mask_a = mask[box[0]]
            tmp_img_c = img[:, box[1], :]
            tmp_mask_c = mask[:, box[1], :]
            tmp_img_s = img[:, :, box[2]]
            tmp_mask_s = mask[:, :, box[2]]
            img_a = transform.resize(tmp_img_a, [sample_size, sample_size], mode='edge', preserve_range='True')
            mask_a = transform.resize(tmp_mask_a, [sample_size, sample_size], mode='edge', preserve_range='True')
            img_c = transform.resize(tmp_img_c, [sample_size, sample_size], mode='edge', preserve_range='True')
            mask_c = transform.resize(tmp_mask_c, [sample_size, sample_size], mode='edge', preserve_range='True')
            img_s = transform.resize(tmp_img_s, [sample_size, sample_size], mode='edge', preserve_range='True')
            mask_s = transform.resize(tmp_mask_s, [sample_size, sample_size], mode='edge', preserve_range='True')
            img_2D[3*i] = img_a
            img_2D[3*i + 1] = img_c
            img_2D[3*i + 2] = img_s
            mask_2D[3*i] = mask_a
            mask_2D[3*i + 1] = mask_c
            mask_2D[3*i + 2] = mask_s
        except:
            print ('-----------error here----------')
    nii_img = nib.Nifti1Image(img_2D, affine = np.eye(4))
    nii_mask = nib.Nifti1Image(mask_2D, affine = np.eye(4))    
    nib.save(nii_img, img_save)
    nib.save(nii_mask, mask_save)
    

def get_2Dnorm_folder(pbb_root, img_root, num_patch, img_save_root, mask_save_root):
    img_list = os.listdir(img_root)
    for i in range(len(img_list)):
        print (i, len(img_list))
        item = img_list[i].replace('.nii.gz', '')
        pbb_path = pbb_root + '/' + item + '_pbb.npy'
        img_path = img_root + '/' + img_list[i]
        img_save = img_save_root + '/' + img_list[i]
        mask_save = mask_save_root + '/' + img_list[i]
        if os.path.exists(img_save):
            print (img_save, ' is existed !')
        else:    
            get_2Dnorm(pbb_path, img_path, num_patch, img_save, mask_save)
        #break

def axial_multislice_npy(pbb_path, img_path, num_patch, num_slice, img_save, mask_save, sample_size = 256): # have not test yet
    pbb_path = '/nfs/masi/NLST/DSB_File/diag/bbox/218866time2000_pbb.npy'
    img_path = '/nfs/masi/NLST/DSB_File/diag/prep/218866time1999_clean.npy'
    lastID = pbb_path.split('/')[-1][:-8]
    currentID = img_path.split('/')[-1][:-10]
    ref_img_path = img_path.replace(currentID, lastID)
    print(ref_img_path)
    data0 = np.load(img_path)
    data1 = np.load(ref_img_path)
    coff = 1.0 * data0.shape[1] / data1.shape[1]
    print(data0.shape, data1.shape)
    print(coff)
    pbb1 = np.load(pbb_path)
    pbb1 = pbb1[np.argsort(-pbb1[:, 0])]
    bboxes = [pbb1[0]]
    print(pbb1[0])
    index0 = 257 * coff
    print(int(index0))


def subj_multislice_npy(pbb_path, img_path, num_patch, num_slice, img_save, mask_save, sample_size = 256):
    '''
    The difference between subj_multislice_npy and ori_multislice_npy is subj_multislice_npy use the last img_path as reference
    '''
    def average_neartwo(img, num_slices):
        l = img.shape[0]
        new_img = np.zeros((num_slices, img.shape[1], img.shape[2]))
        for i in range(l // 2):
            tmp_data = np.stack([img[2 * i], img[2 * i + 1]])
            new_img[i] = np.mean(tmp_data, axis=0)
        for i in range(l//2, num_slices):
            new_img[i] = new_img[l//2 - 1]
        return new_img
    lastID = pbb_path.split('/')[-1][:-8]
    currentID = img_path.split('/')[-1][:-10]
    ref_img_path = img_path.replace(currentID, lastID)
    print(currentID, lastID)
    data0 = np.load(img_path)[0]
    data1 = np.load(ref_img_path)[0]
    pbb = np.load(pbb_path)
    pbb = pbb[pbb[:, 0] > -1.5]  # here has changed
    pbb = nms(pbb, 0.05)
    coff = 1.0 * np.array(data0.shape) / np.array(data1.shape)

    img_2D = np.zeros((3 * num_patch * num_slice, sample_size, sample_size), dtype=np.float)
    mask_2D = np.zeros((3 * num_patch * num_slice, sample_size, sample_size), dtype=np.float)
    boxes = pbb[:num_patch]
    for i in range(len(boxes)):
        box = pbb[i][1:]
        box[:3] = np.array(box[:3]) * coff
        box = box.astype('int')

        tmp_img_a = data0[max(0, box[0] - num_slice): max(0, box[0] - num_slice) + 2 * num_slice]
        tmp_mask_a = np.zeros(tmp_img_a.shape)
        tmp_mask_a[:, max(0, box[1] - box[3]): box[1] + box[3], max(0, box[2] - box[3]): box[2] + box[3]] = 1

        tmp_img_c = data0[:, max(0, box[1] - num_slice): max(0, box[1] - num_slice) + 2 * num_slice, :]
        tmp_mask_c = np.zeros(tmp_img_c.shape)
        tmp_mask_c[max(0, box[0] - box[3]): box[0] + box[3], :, max(0, box[2] - box[3]): box[2] + box[3]] = 1

        tmp_img_s = data0[:, :, max(0, box[2] - num_slice): max(0, box[2] - num_slice) + 2 * num_slice]
        tmp_mask_s = np.zeros(tmp_img_s.shape)
        tmp_mask_s[max(0, box[0] - box[3]): box[0] + box[3], max(0, box[1] - box[3]): box[1] + box[3], :] = 1

        tmp_img_c = np.transpose(tmp_img_c, (1, 0, 2))
        tmp_mask_c = np.transpose(tmp_mask_c, (1, 0, 2))
        tmp_img_s = np.transpose(tmp_img_s, (2, 0, 1))
        tmp_mask_s = np.transpose(tmp_mask_s, (2, 0, 1))

        tmp_img_a = average_neartwo(tmp_img_a, num_slice)
        tmp_img_c = average_neartwo(tmp_img_c, num_slice)
        tmp_img_s = average_neartwo(tmp_img_s, num_slice)

        tmp_mask_a = average_neartwo(tmp_mask_a, num_slice)
        tmp_mask_c = average_neartwo(tmp_mask_c,num_slice)
        tmp_mask_s = average_neartwo(tmp_mask_s,num_slice)

        #print(tmp_img_a.shape, tmp_mask_a.shape,
        #      tmp_img_c.shape, tmp_mask_c.shape,
        #      tmp_img_s.shape, tmp_mask_s.shape)

        img_a = np.zeros([num_slice, sample_size, sample_size], dtype=np.float)
        mask_a = np.zeros([num_slice, sample_size, sample_size], dtype=np.float)
        img_c = np.zeros([num_slice, sample_size, sample_size], dtype=np.float)
        mask_c = np.zeros([num_slice, sample_size, sample_size], dtype=np.float)
        img_s = np.zeros([num_slice, sample_size, sample_size], dtype=np.float)
        mask_s = np.zeros([num_slice, sample_size, sample_size], dtype=np.float)

        for j in range(num_slice):
            # try:
            img_a[j] = transform.resize(tmp_img_a[j], [sample_size, sample_size], mode='edge', preserve_range='True')
            mask_a[j] = transform.resize(tmp_mask_a[j], [sample_size, sample_size], mode='edge', preserve_range='True')
            img_c[j] = transform.resize(tmp_img_c[j], [sample_size, sample_size], mode='edge', preserve_range='True')
            mask_c[j] = transform.resize(tmp_mask_c[j], [sample_size, sample_size], mode='edge', preserve_range='True')
            img_s[j] = transform.resize(tmp_img_s[j], [sample_size, sample_size], mode='edge', preserve_range='True')
            mask_s[j] = transform.resize(tmp_mask_s[j], [sample_size, sample_size], mode='edge', preserve_range='True')
        #print(3 * i * num_slice, (3 * i + 1) * num_slice)
        img_2D[3 * i * num_slice: (3 * i + 1) * num_slice] = img_a
        img_2D[(3 * i + 1) * num_slice: (3 * i + 2) * num_slice] = img_c
        img_2D[(3 * i + 2) * num_slice: (3 * i + 3) * num_slice] = img_s
        mask_2D[3 * i * num_slice: (3 * i + 1) * num_slice] = mask_a
        mask_2D[(3 * i + 1) * num_slice:(3 * i + 2) * num_slice] = mask_c
        mask_2D[(3 * i + 2) * num_slice: (3 * i + 3) * num_slice] = mask_s

    if img_save != None and mask_save != None:
        nii_img = nib.Nifti1Image(img_2D, affine = np.eye(4))
        nii_mask = nib.Nifti1Image(mask_2D, affine = np.eye(4))
        nib.save(nii_img, img_save)
        nib.save(nii_mask, mask_save)

def subj_reg_multislice_npy(pbb_path, img_path, num_patch, num_slice, img_save, mask_save):
    '''
    The difference between subj_multislice_npy and ori_multislice_npy is subj_multislice_npy use the last img_path as reference
    '''
    def average_neartwo(img, num_slices):
        l = img.shape[0]
        new_img = np.zeros((num_slices, img.shape[1], img.shape[2]))
        for i in range(l // 2):
            tmp_data = np.stack([img[2 * i], img[2 * i + 1]])
            new_img[i] = np.mean(tmp_data, axis=0)
        for i in range(l//2, num_slices):
            new_img[i] = new_img[l//2 - 1]
        return new_img

    def average_nearfour(img, num_slices):
        l = img.shape[0]
        assert l // 4 + 1 >= num_slices
        assert l // 4 <= num_slices
        new_img = np.zeros((num_slices, img.shape[1], img.shape[2]))
        for i in range(l // 4):
            tmp_data = np.stack([img[4 * i], img[2 * i + 3]])
            new_img[i] = np.mean(tmp_data, axis=0)
        for i in range(l // 4, num_slices):
            new_img[i] = new_img[l // 4 - 1]
        return new_img

    lastID = pbb_path.split('/')[-1][:-8]
    currentID = img_path.split('/')[-1][:-10]
    #ref_img_path = img_path.replace(currentID, lastID)
    print(currentID, lastID)
    data0 = np.load(img_path)[0]
    assert data0.shape == (325, 254, 380)
    pbb = np.load(pbb_path)
    pbb = pbb[pbb[:, 0] > -1.5]  # here has changed
    pbb = nms(pbb, 0.05)
#    coff = 1.0 * np.array(data0.shape) / np.array(data1.shape)

    img_2D = np.zeros(( num_patch * num_slice, data0.shape[1], data0.shape[2]), dtype=np.float)
    mask_2D = np.zeros(( num_patch * num_slice, data0.shape[1], data0.shape[2]), dtype=np.float)
    boxes = pbb[:num_patch]
    for i in range(len(boxes)):
        box = pbb[i][1:]
        #box[:3] = np.array(box[:3]) * coff
        box = box.astype('int')
        bsize = max(12, box[3])
        bsize = min(32, bsize)

        tmp_img_a = data0[max(0, box[0] - num_slice): max(0, box[0] - num_slice) + 2 * num_slice]
        tmp_mask_a = np.zeros(tmp_img_a.shape)
        tmp_mask_a[:, max(0, box[1] - bsize): box[1] + bsize, max(0, box[2] - bsize): box[2] + bsize] = 1

        tmp_img_a = average_neartwo(tmp_img_a, num_slice)
        #tmp_img_c = average_neartwo(tmp_img_c, num_slice)
        #tmp_img_s = average_neartwo(tmp_img_s, num_slice)

        tmp_mask_a = average_neartwo(tmp_mask_a, num_slice)
        #tmp_mask_c = average_neartwo(tmp_mask_c,num_slice)
        #tmp_mask_s = average_neartwo(tmp_mask_s,num_slice)

        img_2D[i * num_slice: ( i + 1) * num_slice] = tmp_img_a
        #img_2D[(3 * i + 1) * num_slice: (3 * i + 2) * num_slice] = img_c
        #img_2D[(3 * i + 2) * num_slice: (3 * i + 3) * num_slice] = img_s
        mask_2D[ i * num_slice: ( i + 1) * num_slice] = tmp_mask_a
        #mask_2D[(3 * i + 1) * num_slice:(3 * i + 2) * num_slice] = mask_c
        #mask_2D[(3 * i + 2) * num_slice: (3 * i + 3) * num_slice] = mask_s

    if img_save != None and mask_save != None:
        nii_img = nib.Nifti1Image(img_2D, affine = np.eye(4))
        nii_mask = nib.Nifti1Image(mask_2D, affine = np.eye(4))
        nib.save(nii_img, img_save)
        nib.save(nii_mask, mask_save)


def subj_reg_multislice4_npy(pbb_path, img_path, num_patch, num_slice, img_save, mask_save):
    '''
    The difference between subj_multislice_npy and ori_multislice_npy is subj_multislice_npy use the last img_path as reference
    '''


    def average_nearfour(img, num_slices):
        l = img.shape[0]
        #assert l // 4 + 1 >= num_slices
        #assert l // 4 <= num_slices
        new_img = np.zeros((num_slices, img.shape[1], img.shape[2]))
        for i in range(l // 4):
            tmp_data = np.stack([img[4 * i], img[2 * i + 3]])
            new_img[i] = np.mean(tmp_data, axis=0)
        for i in range(l // 4, num_slices):
            new_img[i] = new_img[l // 4 - 1]
        return new_img

    lastID = pbb_path.split('/')[-1][:-8]
    currentID = img_path.split('/')[-1][:-10]
    # ref_img_path = img_path.replace(currentID, lastID)
    print(currentID, lastID)
    data0 = np.load(img_path)[0]
    assert data0.shape == (325, 254, 380)
    pbb = np.load(pbb_path)
    pbb = pbb[pbb[:, 0] > -1.5]  # here has changed
    pbb = nms(pbb, 0.05)
    #    coff = 1.0 * np.array(data0.shape) / np.array(data1.shape)

    img_2D = np.zeros((num_patch * num_slice, data0.shape[1], data0.shape[2]), dtype=np.float)
    mask_2D = np.zeros((num_patch * num_slice, data0.shape[1], data0.shape[2]), dtype=np.float)
    boxes = pbb[:num_patch]
    for i in range(len(boxes)):
        box = pbb[i][1:]
        # box[:3] = np.array(box[:3]) * coff
        box = box.astype('int')
        bsize = max(12, box[3])
        bsize = min(32, bsize)

        tmp_img_a = data0[max(0, box[0] - 2 * num_slice): max(0, box[0] - 2 * num_slice) + 4 * num_slice]
        tmp_mask_a = np.zeros(tmp_img_a.shape)
        tmp_mask_a[:, max(0, box[1] - bsize): box[1] + bsize, max(0, box[2] - bsize): box[2] + bsize] = 1

        tmp_img_a = average_nearfour(tmp_img_a, num_slice)

        tmp_mask_a = average_nearfour(tmp_mask_a, num_slice)

        img_2D[i * num_slice: (i + 1) * num_slice] = tmp_img_a
        # img_2D[(3 * i + 1) * num_slice: (3 * i + 2) * num_slice] = img_c
        # img_2D[(3 * i + 2) * num_slice: (3 * i + 3) * num_slice] = img_s
        mask_2D[i * num_slice: (i + 1) * num_slice] = tmp_mask_a
        # mask_2D[(3 * i + 1) * num_slice:(3 * i + 2) * num_slice] = mask_c
        # mask_2D[(3 * i + 2) * num_slice: (3 * i + 3) * num_slice] = mask_s

    if img_save != None and mask_save != None:
        nii_img = nib.Nifti1Image(img_2D, affine=np.eye(4))
        nii_mask = nib.Nifti1Image(mask_2D, affine=np.eye(4))
        nib.save(nii_img, img_save)
        nib.save(nii_mask, mask_save)

def subj_reg_folder(pbb_root, img_root, num_patch, num_slice,  img_save_root, mask_save_root):
    img_list = os.listdir(img_root)
    for i in range(0, len(img_list)):
        # if i > 2: break
        if i % 5 == 0: print(i, len(img_list))
        if img_list[i][-9:] != 'clean.npy': continue

        subj = re.split("[time]", img_list[i])[0]
        paths = glob(os.path.join(pbb_root, subj + '*'))
        print(img_list[i], paths)
        if len(paths) == 0: continue
        #assert len(paths) == 1
        paths = sorted(paths, reverse=True)
        pbb_path = paths[0]

        # item = img_list[i].replace('_clean.npy', '')
        # pbb_path = pbb_root + '/' + item + '_pbb.npy'
        img_path = img_root + '/' + img_list[i]
        img_save = img_save_root + '/' + img_list[i].replace('_clean.npy', '.nii.gz')
        mask_save = mask_save_root + '/' + img_list[i].replace('_clean.npy', '.nii.gz')
        # if not os.path.exists(pbb_path):
        #     continue
        #
        # if os.path.exists(img_save):
        #     continue
        subj_reg_multislice4_npy(pbb_path, img_path, num_patch, num_slice, img_save, mask_save)


def ori_multislice_npy(pbb_path, img_path, num_patch, num_slice, img_save, mask_save, sample_size = 256, resize = False):
    img = np.load(img_path)
    assert len(img.shape) == 4
    img = img[0]
    pbb = np.load(pbb_path)
    pbb = pbb[pbb[:,0]>-1.5]  # here has changed
    pbb = nms(pbb,0.05)
    boxes = pbb[:num_patch]
    img_2D = np.zeros((3 * num_patch * num_slice, sample_size, sample_size), dtype = np.float)
    mask_2D = np.zeros((3 * num_patch * num_slice, sample_size, sample_size), dtype = np.float)
    size = img.shape
    #print (img.shape)
    for i in range(len(boxes)):
        box = boxes[i].astype('int')[1:]
        #print (box)
        tmp_img_a = img[max(0, box[0] - num_slice // 2): max(0, box[0] - num_slice // 2) + num_slice]
        tmp_mask_a = np.zeros(tmp_img_a.shape)
        tmp_mask_a[:, max(0,box[1] - box[3]): box[1] + box[3], max(0,box[2] - box[3]): box[2] + box[3]] = 1
        tmp_img_c = img[:, max(0, box[1] - num_slice // 2): max(0, box[1] - num_slice // 2) + num_slice, :]
        tmp_mask_c = np.zeros(tmp_img_c.shape)
        tmp_mask_c[max(0,box[0] - box[3]): box[0] + box[3],:, max(0,box[2] - box[3]): box[2] + box[3]] = 1
        #nii_img = nib.Nifti1Image(tmp_img_c, affine = np.eye(4))    
        #nib.save(nii_img, '/nfs/masi/gaor2/data/MCL/multiS_norm/tmp_img_c.nii.gz')
        
        tmp_img_s = img[:, :, max(0, box[2] - num_slice // 2): max(0, box[2] - num_slice // 2) + num_slice]
        tmp_mask_s = np.zeros(tmp_img_s.shape)
        tmp_mask_s[max(0,box[0] - box[3]): box[0] + box[3], max(0,box[1] - box[3]): box[1] + box[3], :] = 1
        #for j in range(len(num_slice)):
        #print (tmp_img_a.shape)
#         for k in range(3):
#         for j in range(num_slice):
#             img_2D[i * (0 + j)] = transform.resize(tmp_img_a[j], [sample_size, sample_size], mode='edge', preserve_range='True')
#             mask_2D[k * i + j] = transform.resize(tmp_mask_a[j], [ sample_size, sample_size], mode='edge', preserve_range='True')
        #print (tmp_img_a.shape, tmp_img_c.shape, tmp_img_s.shape)
        tmp_img_c = np.transpose(tmp_img_c, (1,0,2))
        tmp_mask_c = np.transpose(tmp_mask_c, (1,0,2))
        tmp_img_s = np.transpose(tmp_img_s, (2,0,1))
        tmp_mask_s = np.transpose(tmp_mask_s, (2,0,1))
        img_a = np.zeros([num_slice, sample_size, sample_size], dtype = np.float)
        mask_a= np.zeros([num_slice, sample_size, sample_size], dtype = np.float)
        img_c = np.zeros([num_slice, sample_size, sample_size], dtype = np.float)
        mask_c= np.zeros([num_slice, sample_size, sample_size], dtype = np.float)
        img_s= np.zeros([num_slice, sample_size, sample_size], dtype = np.float)
        mask_s= np.zeros([num_slice, sample_size, sample_size], dtype = np.float)
        #print (tmp_img_a.shape, img_path)
        for j in range(num_slice): 
            try:
                img_a[j] = transform.resize(tmp_img_a[j], [sample_size, sample_size], mode='edge', preserve_range='True')
                mask_a[j] = transform.resize(tmp_mask_a[j], [sample_size, sample_size], mode='edge', preserve_range='True')
                img_c[j] = transform.resize(tmp_img_c[j], [sample_size, sample_size], mode='edge', preserve_range='True')
                mask_c[j] = transform.resize(tmp_mask_c[j], [sample_size, sample_size], mode='edge', preserve_range='True')
                img_s[j] = transform.resize(tmp_img_s[j], [ sample_size, sample_size], mode='edge', preserve_range='True')
                mask_s[j] = transform.resize(tmp_mask_s[j], [ sample_size, sample_size], mode='edge', preserve_range='True')
            except:
                print ('nodule error')
            
        img_2D[3* i * num_slice: (3* i + 1)* num_slice] = img_a
        img_2D[(3* i + 1)* num_slice: (3* i + 2) * num_slice] = img_c
        img_2D[(3* i + 2)* num_slice: (3* i + 3) * num_slice] = img_s
        mask_2D[3* i * num_slice: (3* i + 1)* num_slice] = mask_a
        mask_2D[(3* i + 1)* num_slice:( 3* i + 2) * num_slice] = mask_c
        mask_2D[(3* i + 2)* num_slice: (3* i + 3) * num_slice] = mask_s
        #print (np.max(mask_a), np.max(mask_c), np.max(mask_s))
        #print (np.sum(mask_2D))
        #except:
        #    print ('error here-----------')
    # if os.path.exists(img_save):
    #     print (img_save, ' is existed')
    if img_save != None and mask_save != None:
        nii_img = nib.Nifti1Image(img_2D, affine = np.eye(4))
        nii_mask = nib.Nifti1Image(mask_2D, affine = np.eye(4))
        nib.save(nii_img, img_save)
        nib.save(nii_mask, mask_save)
            
def get_2Dnorm_npy(pbb_path, img_path, num_patch, img_save, mask_save, sample_size = 256):
    img = np.load(img_path)
    assert len(img.shape) == 4
    img = img[0]
    pbb = np.load(pbb_path)
    pbb = pbb[pbb[:,0]>-1.5]  # here has changed
    pbb = nms(pbb,0.05)
    boxes = pbb[:num_patch]
    img_2D = np.zeros((3 * num_patch, sample_size, sample_size), dtype = np.uint8)
    mask_2D = np.zeros((3 * num_patch, sample_size, sample_size), dtype = np.uint8)
    for i in range(len(boxes)):
        box = boxes[i].astype('int')[1:]
        try:
            tmp_img_a = img[box[0]]
            tmp_mask_a = np.zeros(tmp_img_a.shape)
            tmp_mask_a[max(0,box[1] - 32): box[1] + 32, max(0,box[2] - 32): box[2] + 32] = 1
            tmp_img_c = img[:, box[1], :]
            tmp_mask_c = np.zeros(tmp_img_c.shape)
            tmp_mask_c[max(0,box[0] - 32): box[0] + 32, max(0,box[2] - 32): box[2] + 32] = 1
            tmp_img_s = img[:, :, box[2]]
            tmp_mask_s = np.zeros(tmp_img_s.shape)
            tmp_mask_s[max(0,box[0] - 32): box[0] + 32, max(0,box[1] - 32): box[1] + 32] = 1
            img_a = transform.resize(tmp_img_a, [sample_size, sample_size], mode='edge', preserve_range='True')
            mask_a = transform.resize(tmp_mask_a, [sample_size, sample_size], mode='edge', preserve_range='True')
            img_c = transform.resize(tmp_img_c, [sample_size, sample_size], mode='edge', preserve_range='True')
            mask_c = transform.resize(tmp_mask_c, [sample_size, sample_size], mode='edge', preserve_range='True')
            img_s = transform.resize(tmp_img_s, [sample_size, sample_size], mode='edge', preserve_range='True')
            mask_s = transform.resize(tmp_mask_s, [sample_size, sample_size], mode='edge', preserve_range='True')
            img_2D[3*i] = img_a
            img_2D[3*i + 1] = img_c
            img_2D[3*i + 2] = img_s
            mask_2D[3*i] = mask_a
            mask_2D[3*i + 1] = mask_c
            mask_2D[3*i + 2] = mask_s
        except:
            print ('error here-----------')
        
    nii_img = nib.Nifti1Image(img_2D, affine = np.eye(4))
    nii_mask = nib.Nifti1Image(mask_2D, affine = np.eye(4))    
    nib.save(nii_img, img_save)
    nib.save(nii_mask, mask_save)  
    
def guass_2Dnorm_npy(pbb_path, img_path, num_patch, img_save, mask_save, sample_size = 256):
    img = np.load(img_path)
    assert len(img.shape) == 4
    img = img[0]
    pbb = np.load(pbb_path)
    pbb = pbb[pbb[:,0]>-1.5]  # here has changed
    pbb = nms(pbb,0.05)
    boxes = pbb[:num_patch]
    img_2D = np.zeros(( 3 * num_patch, sample_size, sample_size), dtype = np.uint8)
    mask_2D = np.zeros(( 3 * num_patch, sample_size, sample_size), dtype = np.uint8)
    for i in range(len(boxes)):
        box = boxes[i].astype('int')[1:]
        
        try:
            
            tmp_img_a = img[box[0]]
            coors_a = np.mgrid[:tmp_img_a.shape[0], :tmp_img_a.shape[1]]
            dis_map_a = (coors_a[0] - box[1]) ** 2 + ((coors_a[1] - box[2])) ** 2
            tmp_mask_a = 255 * np.exp(- dis_map_a / 1600.)
            
            tmp_img_c = img[:, box[1], :]
            coors_c = np.mgrid[:tmp_img_c.shape[0], :tmp_img_c.shape[1]]
            dis_map_c = (coors_c[0] - box[0]) ** 2 + ((coors_c[1] - box[2])) ** 2
            tmp_mask_c = 255 * np.exp( - dis_map_c / 1600. )
            
            tmp_img_s = img[:, : ,box[2]]
            coors_s = np.mgrid[:tmp_img_s.shape[0], :tmp_img_s.shape[1]]
            dis_map_s = (coors_s[0] - box[0]) ** 2 + ((coors_s[1] - box[1])) ** 2
            tmp_mask_s = 255 * np.exp( - dis_map_s / 1600.)
            
            img_a = transform.resize(tmp_img_a, [sample_size, sample_size], mode='edge', preserve_range='True')
            mask_a = transform.resize(tmp_mask_a, [sample_size, sample_size], mode='edge', preserve_range='True')
            img_c = transform.resize(tmp_img_c, [sample_size, sample_size], mode='edge', preserve_range='True')
            mask_c = transform.resize(tmp_mask_c, [sample_size, sample_size], mode='edge', preserve_range='True')
            img_s = transform.resize(tmp_img_s, [sample_size, sample_size], mode='edge', preserve_range='True')
            mask_s = transform.resize(tmp_mask_s, [sample_size, sample_size], mode='edge', preserve_range='True')
            
            img_2D[3*i] = img_a
            img_2D[3*i + 1] = img_c
            img_2D[3*i + 2] = img_s
            mask_2D[3*i] = mask_a
            mask_2D[3*i + 1] = mask_c
            mask_2D[3*i + 2] = mask_s

        except:
            print ('error here-----------')
    if True: #not os.path.exists(img_save):  
        nii_img = nib.Nifti1Image(img_2D, affine = np.eye(4))
        nib.save(nii_img, img_save)
    if True: #not os.path.exists(mask_save):  
        nii_mask = nib.Nifti1Image(mask_2D, affine = np.eye(4))    
        nib.save(nii_mask, mask_save)


    
def npy_2Dnorm_folder(pbb_root, img_root, num_patch, num_slice, sample_size, img_save_root, mask_save_root):
    img_list = os.listdir(img_root)
    for i in range(0,len(img_list)):
        #if i > 2: break
        if i % 5 == 0: print (i, len(img_list))
        if img_list[i][-9:] != 'clean.npy': continue
        item = img_list[i].replace('_clean.npy', '')
        pbb_path = pbb_root + '/' + item + '_pbb.npy'
        img_path = img_root + '/' + img_list[i]
        img_save = img_save_root + '/' + img_list[i].replace('_clean.npy', '.nii.gz')
        mask_save = mask_save_root + '/' + img_list[i].replace('_clean.npy', '.nii.gz')
        if not os.path.exists(pbb_path):
            continue
        #get_2Dnorm_npy(pbb_path, img_path, num_patch, img_save, mask_save)
        #guass_2Dnorm_npy(pbb_path, img_path, num_patch, img_save, mask_save)
        if os.path.exists(img_save):
            continue
        ori_multislice_npy(pbb_path, img_path, num_patch, num_slice, img_save, mask_save, sample_size = sample_size)

def npy_2Dnorm_subj(pbb_root, img_root, num_patch, num_slice, sample_size, img_save_root, mask_save_root):
    img_list = os.listdir(img_root)
    for i in range(0,len(img_list)):
        #if i > 10: break

        if i % 5 == 0: print (i, len(img_list))
        if img_list[i][-9:] != 'clean.npy': continue
        img_save = img_save_root + '/' + img_list[i].replace('_clean.npy', '.nii.gz')
        mask_save = mask_save_root + '/' + img_list[i].replace('_clean.npy', '.nii.gz')
        # if not os.path.exists(img_save):
        #     print (i, img_list[i])
        #     break
        # else:
        #     continue
        item = img_list[i].replace('_clean.npy', '')
        #pbb_path = pbb_root + '/' + item + '_pbb.npy'
        img_path = img_root + '/' + img_list[i]

        subj = re.split("[time]", img_list[i])[0]
        paths = glob(os.path.join(pbb_root, subj + '*'))
        print (img_list[i], paths)
        paths = sorted(paths, reverse = True)
        #assert len(paths) == 1
        pbb_path = paths[0]
        #get_2Dnorm_npy(pbb_path, img_path, num_patch, img_save, mask_save)
        #guass_2Dnorm_npy(pbb_path, img_path, num_patch, img_save, mask_save)
        if os.path.exists(img_save):
            continue
        ori_multislice_npy(pbb_path, img_path, num_patch, num_slice, img_save, mask_save, sample_size = sample_size) # this line create multiS_new2
        #subj_multislice_npy(pbb_path, img_path, num_patch, num_slice, img_save, mask_save, sample_size = sample_size) # this line create multiS_new3
        
        
def get_roi(mask_path):
    img_nii = nib.load(mask_path)
    img = img_nii.get_data()
    roi = np.zeros(img.shape, dtype = np.uint8)
    x_list, y_list, z_list = [], [], []
    for i in range(img.shape[0]):
        if np.sum(img[i, :, :]) > 20:
            x_list.append(i)
    for i in range(img.shape[1]):
        if np.sum(img[:, i, :]) > 20:
            y_list.append(i)
    for i in range(img.shape[2]):
        if np.sum(img[:, :, i]) > 20:
            z_list.append(i)
            #roi[:, :, i] = 1
    x_cent, y_cent, z_cent =  (x_list[0] + x_list[-1]) / 2, (y_list[0] + y_list[-1]) / 2, (z_list[0] + z_list[-1]) / 2
    #print ('the center is: ', x_cent, y_cent, z_cent)
    return x_cent, y_cent, z_cent
        
def save_roi_mask(mask_path, save_path):
    center = get_roi(mask_path)
    mask_nii = nib.load(mask_path)

    resol = [mask_nii.header['pixdim'][1], mask_nii.header['pixdim'][2], mask_nii.header['pixdim'][3]]
    lens = [def_size * def_reso / i for i in resol]
    x_begin, x_end = center[0] - lens[0] / 2, center[0] + lens[0] / 2
    y_begin, y_end = center[1] - lens[1] / 2, center[1] + lens[1] / 2
    z_begin, z_end = center[2] - lens[2] / 2, center[2] + lens[2] / 2
    mask_new = mask_nii.get_data()
    x_begin, x_end, y_begin, y_end, z_begin, z_end = max(0, int(x_begin)), int(x_end), max(0, int(y_begin)), int(
        y_end), max(0, int(z_begin)), int(z_end)
    mask_new = mask_new[x_begin: x_end, y_begin: y_end, z_begin: z_end]
    mask_save = transform.resize(mask_new, [def_size, def_size, def_size], mode='edge', preserve_range='True')
    save_nii = nib.Nifti1Image(mask_save, np.array([[-1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])) 
    # np.eye(4)
    nib.save(save_nii, save_path)

def change_affine(data_root, affine):
    data_list = os.listdir(data_root)
    for i in range(len(data_list)):
        print (i, len(data_list), data_list[i])
        img_nii = nib.load(data_root + '/' + data_list[i])
        img = img_nii.get_data()
        save_nii = nib.Nifti1Image(img, affine)
        nib.save(save_nii, data_root + '/' + data_list[i])

def orient2std(data_root, new_root):
    data_list = os.listdir(data_root)
    for i in range(len(data_list)):
        print (i, 'fslreorient2std ' + data_root + '/' + data_list[i] + ' ' + new_root + '/' + data_list[i])
        os.system('fslreorient2std ' + data_root + '/' + data_list[i] + ' ' + new_root + '/' + data_list[i])
        
def get_padmask(kaggle_mask_root, save_root):
    ori_mask_list = os.listdir(kaggle_mask_root)
    for i in range(len(ori_mask_list)):
        print (i, len(ori_mask_list), ori_mask_list[i])
        mask_nii = nib.load(kaggle_mask_root + '/' + ori_mask_list[i])
        new_size = list(mask_nii.shape)
        new_size = [int(i / 1.5) for i in new_size]

        new_mask = transform.resize(mask_nii.get_data(), new_size, mode='edge', preserve_range='True')
        mask_save = np.zeros([200, 200, 200])
        x_begin, x_end = 100 - new_size[0] / 2, 100 + new_size[0] / 2
        y_begin, y_end = 100 - new_size[1] / 2, 100 + new_size[1] / 2
        z_begin, z_end = 100 - new_size[2] / 2, 100 + new_size[2] / 2
        x_begin, x_end, y_begin, y_end, z_begin, z_end = max(0, int(x_begin)), int(x_end), max(0, int(y_begin)), int(
            y_end), max(0, int(z_begin)), int(z_end)

        mask_save[x_begin: x_end, y_begin: y_end, z_begin: z_end] = new_mask[0: min(200, x_end - x_begin), 0: min(y_end - y_begin, 200), 0: min(200, z_end-  z_begin)]
        affine = np.array([[0,0,-1,0], [0,-1,0,0], [1,0,0,0], [0,0,0,1]])
        save_nii = nib.Nifti1Image(mask_save, affine)

        nib.save(save_nii, save_root + '/' + ori_mask_list[i])
        #break

def slurm_mask(data_root, script_root, log_root, save_mask_root, make_dir = True):
    subj_list = os.listdir(data_root)
    for i in range(len(subj_list)):
        print (i, subj_list[i])
        subj_path = data_root + '/' + subj_list[i]
        sess_list = os.listdir(subj_path)
        for j in range(len(sess_list)):
            nii_path = subj_path + '/' + sess_list[j] + '/' + subj_list[i] + 'time' + sess_list[j] + '.nii.gz'
            mask_path = save_mask_root + '/' + subj_list[i] + 'time' + sess_list[j] + '.nii.gz'
            slum_path = script_root  + '/' + subj_list[i] + 'time' + sess_list[j] + '.sh'
            log_path = log_root + '/' + subj_list[i] + 'time' + sess_list[j] + '.txt'

            # if make_dir:
            #     mkdir(save_mask_root + '/' + subj_list[i] + '/' + sess_list[j])
            #     mkdir(script_root + '/' + subj_list[i] + '/' + sess_list[j])
            #     mkdir(log_root + '/' + subj_list[i] + '/' + sess_list[j])

            f = open(slum_path, 'w')
            f.write('#!/bin/bash\n')
            f.write('#SBATCH --job-name=' + subj_list[i] + 'time' + sess_list[j] + '\n')
            f.write('#SBATCH --output=' + log_path + '\n')
            f.write('#SBATCH --ntasks=1\n')
            f.write('#SBATCH --time=01:00:00\n')
            f.write('#SBATCH --mem-per-cpu=11G\n')
            f.write('/home-nfs2/local/VANDERBILT/gaor2/anaconda3/envs/python37/bin/python /home-nfs2/local/VANDERBILT/gaor2/code/RNNLung/func/tools/seg_lung.py --ori ' + nii_path + ' --out ' + mask_path + '\n')
            f.write('\ndate')
            f.close()
            #break
        #break

def run_mask_slurm(slurm_root, save_root):
    slurm_list = os.listdir(slurm_root)
    exist_list = os.listdir(save_root)
    for i in range(len(slurm_list)):
        if slurm_list[i].replace('sh', 'nii.gz') not in exist_list:
            print (slurm_list[i].replace('sh', 'nii.gz'))
            os.system('sbatch ' + slurm_root + '/' + slurm_list[i])
        #else:
        #    print (slurm_list[i].replace('sh', 'nii.gz') + ' is existed ---------------------------------------')
        

# ----------------------- The following a few functions for transfer pbb ---------------

def align_pbb_update(mov_pbb, old_align_pbb, ref_pbb, save_new_pbb):
    pbb0 = np.load(mov_pbb)
    old_align_pbb0 = np.load(old_align_pbb)
    pbb1 = np.load(ref_pbb)
    pbb1 = pbb1[pbb1[:, 0] > -1]
    pbb1 = nms(pbb1, 0.05)
    pbb0 = pbb0[pbb0[:, 0] > -1]
    pbb0 = nms(pbb0, 0.05)
    print (old_align_pbb0.shape, pbb1.shape)
    assert len(old_align_pbb0) == len(pbb1)
    new_pbb0 = np.zeros(pbb1.shape)
    for i in range(len(pbb1)):
        tmp_i_coord = pbb1[i][1:4]
        min_dis = 1000
        min_indx = -1
        for j in range(len(pbb0)):
            tmp_j_coord = pbb0[j][1:4]
            if np.linalg.norm(tmp_i_coord - tmp_j_coord) < min_dis:
                min_dis = np.linalg.norm(tmp_i_coord - tmp_j_coord)
                min_idx = j
        if min_dis > 18:
            new_pbb0[i] = old_align_pbb0[i]
        else:
            new_pbb0[i] = pbb0[min_idx]
    np.save(save_new_pbb, new_pbb0)

def align_pbb( mov_npy, ref_npy,ref_pbb, save_mov_pbb):
    
    '''
    Made a update for this function at 20201030, see align_pbb_update. 
    '''
    
    npy_mov = np.load(mov_npy)
    npy_ref = np.load(ref_npy)
    pbb_ref = np.load(ref_pbb)
    assert len(npy_mov.shape) == 4
    assert len(npy_ref.shape) == 4
    pbb_ref = pbb_ref[pbb_ref[:, 0] > -1]
    pbb_ref = nms(pbb_ref, 0.05)
    new_pbb = pbb_ref.copy()
    print (new_pbb.shape)
    scale_ratio = np.array(npy_mov.shape[1:]) / np.array(npy_ref.shape[1:])
    for i in range(len(new_pbb)):
        new_pbb[i, 1:4] = pbb_ref[i, 1:4] * scale_ratio
    print (new_pbb.shape)
    np.save(save_mov_pbb,new_pbb )


def get_transfer_csv(pbb_root, save_csv_path):
    pbb_list = os.listdir(pbb_root)
    pbb_list = [i for i in pbb_list if i[-7:] == 'pbb.npy']
    subj_sess = {}
    for i in range(len(pbb_list)):

        tmp_vec = re.split('[time_]', pbb_list[i])
        subj = tmp_vec[0]
        if subj not in subj_sess.keys():
            subj_sess[subj] = []
        subj_sess[subj].append(pbb_list[i])
    mov_list, ref_list = [], []
    for i in range(len(pbb_list)):
        tmp_vec = re.split('[time_]', pbb_list[i])
        subj = tmp_vec[0]
        mov_list.append(pbb_list[i])
        ref_list.append(max(subj_sess[subj]))
    data = pd.DataFrame()
    data['mov'] = mov_list
    data['ref'] = ref_list
    data.to_csv(save_csv_path, index=False)
    
def get_transfer_from_csv(csv_path, save_csv_path):
    df = pd.read_csv(csv_path)
    df = df.loc[df['source'] == 'mcl']
    df = df.query('no_prep == 0')
    subj_sess = {}
    for i, item in df.iterrows():
        subj = item['item'].split('t')[0]
        if subj not in subj_sess.keys():
            subj_sess[subj] = []
        subj_sess[subj].append(item['item'] + '_pbb.npy')
        
    mov_list, ref_list = [], []
    
    for i, item in df.iterrows():
        subj = item['item'].split('t')[0]
        mov_list.append(item['item'] + '_pbb.npy')
        ref_list.append(max(subj_sess[subj]))
    data = pd.DataFrame()
    data['mov'] = mov_list
    data['ref'] = ref_list
    data.to_csv(save_csv_path, index=False)
    
def align_pbb_fold_update(pbb_root, old_align_root, transfer_csv, new_pbb_root):
    df = pd.read_csv(transfer_csv)
    for i, item in df.iterrows():
        #if i > 1: break
        if (i % 10) == 0: print (i)
#         mov_id = item['mov'] #.replace('pbb', 'clean')
#         ref_id = item['ref'] #.replace('pbb', 'clean')
#         mov_npy = npy_root + '/' + mov_id
#         ref_npy = npy_root + '/' + ref_id
        ref_pbb = pbb_root + '/' + item['ref']
        mov_pbb = pbb_root + '/' + item['mov']
        old_align_pbb = old_align_root + '/' + item['mov']
        save_mov_pbb = new_pbb_root + '/' + item['mov']
        if os.path.exists(save_mov_pbb):
            print (save_mov_pbb + ' is existed')
            continue
        if item['ref'] == item['mov']:
            os.system('cp ' + ref_pbb + ' ' + save_mov_pbb)
        else:
            align_pbb_update(mov_pbb, old_align_pbb, ref_pbb, save_mov_pbb)
            #align_pbb(mov_npy, ref_npy,ref_pbb, save_mov_pbb)
    
def align_pbb_fold(pbb_root, npy_root, transfer_csv, new_pbb_root):
    
    '''
    Made a update for this function at 20201030, see align_pbb_fold_update. 
    '''
    
    df = pd.read_csv(transfer_csv)
    for i, item in df.iterrows():
        #if i > 1: break
        if (i % 10) == 0: print (i)
        mov_id = item['mov'].replace('pbb', 'clean')
        ref_id = item['ref'].replace('pbb', 'clean')
        mov_npy = npy_root + '/' + mov_id
        ref_npy = npy_root + '/' + ref_id
        ref_pbb = pbb_root + '/' + item['ref']
        save_mov_pbb = new_pbb_root + '/' + item['mov']
        if os.path.exists(save_mov_pbb):
            print (save_mov_pbb + ' is existed')
            continue
        if mov_id == ref_id:
            os.system('cp ' + ref_pbb + ' ' + save_mov_pbb)
        else:
            align_pbb(mov_npy, ref_npy,ref_pbb, save_mov_pbb)


        