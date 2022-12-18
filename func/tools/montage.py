import numpy as np
import nibabel as nb
import os
from PIL import Image
import pandas as pd

def single(img_path, img_save, img_size):
    img_nii = nb.load(img_path).get_data()
    print (img_nii.shape)
    new_img = np.zeros((4 * img_size, 4 * img_size))
    for i in range(4):
        for j in range(4):
            if i * 4 + j >= 15:
                break
            new_img[img_size * i : img_size* (i + 1), img_size * j : img_size* (j + 1)] = img_nii[i* 4 + j] 
    img = Image.fromarray(new_img)
    img = img.convert('RGB')
    img.save(img_save)
    
def single_folder(img_root, save_root, img_size):
    data_list = os.listdir(img_root)
    for i in range(len(data_list)):
        single(img_root + '/' + data_list[i], save_root + '/' + data_list[i].replace('.nii.gz', '.png'), img_size)
    
def single_multi(img_path, mask_path, img_save, sample_size):
    img_nii = nb.load(img_path).get_data()
    mask_nii = nb.load(mask_path).get_data()
    width = 8
    new_img = np.zeros((width * sample_size, width * sample_size))
    for i in range(width):
        for j in range(width):
            if i * width + j >= 60:
                break
            tmp_img = img_nii[i* width + j] *(0.5 + mask_nii[i* width + j] * 0.6)
            if (i* width + j) % 3 != 0:
                tmp_img = np.rot90(tmp_img, 2)
            new_img[sample_size * i : sample_size* (i + 1), sample_size * j : sample_size* (j + 1)] = tmp_img #img_nii[i* 4 + j] *(0.5 + mask_nii[i* 4 + j] * 0.6)
            #print (np.max(mask_nii))
    img = Image.fromarray(new_img)
    img = img.convert('RGB')
    img.save(img_save)
    
def single_slice(img_path, mask_path, img_save, sample_size):
    img_nii = nb.load(img_path).get_data()
    mask_nii = nb.load(mask_path).get_data()
    new_img = np.zeros((5 * sample_size, 5 * sample_size))
    for i in range(5):
        for j in range(5):
            #if i * 2 + j >= 4:
            #    break
            tmp_img = img_nii[i* 5 + j] *(0.5 + mask_nii[i* 5 + j]  * 0.6)
            
            new_img[sample_size * i : sample_size* (i + 1), sample_size * j : sample_size* (j + 1)] = tmp_img 
    img = Image.fromarray(new_img)
    img = img.convert('RGB')
    img.save(img_save)

def reg_slice(img_path, mask_path, img_save, sample_size):
    img_nii = nb.load(img_path).get_data()
    mask_nii = nb.load(mask_path).get_data()
    new_img = np.zeros((5 * sample_size[0], 5 * sample_size[1]))
    for i in range(5):
        for j in range(5):
            #if i * 2 + j >= 4:
            #    break
            tmp_img = img_nii[i* 5 + j] *(0.5 + mask_nii[i* 5 + j]  * 0.6)

            new_img[sample_size[0] * i : sample_size[0]* (i + 1), sample_size[1] * j : sample_size[1] * (j + 1)] = tmp_img
    img = Image.fromarray(new_img)
    img = img.convert('RGB')
    img.save(img_save)

df = pd.read_csv('/nfs/masi/gaor2/data/Cotrain/nlst/nlst_pos1yr.csv')
df = df.loc[df['gt'] == 1]
needed_list = df['item'].tolist()
needed_list = [i[:6] for i in needed_list]

def reg_folder_multi(img_root, simple_size):
    img_list = os.listdir(img_root)
    for i in range(len(img_list)):
        #print (i)
        img_path = img_root + '/' + img_list[i]
        mask_path = img_path.replace('img', 'mask')
        save_path = img_path.replace('img', 'montage').replace('.nii.gz', '.jpeg')
        #single_multi(img_path, mask_path, save_path, simple_size)
        reg_slice(img_path, mask_path, save_path, simple_size)
    
def folder_multi(img_root, simple_size):
    img_list = os.listdir(img_root)
    for i in range(len(img_list)):
        #print (i)
        img_path = img_root + '/' + img_list[i]
        mask_path = img_path.replace('img', 'mask')
        save_path = img_path.replace('img', 'montage').replace('.nii.gz', '.jpeg')
        #single_multi(img_path, mask_path, save_path, simple_size)
        single_slice(img_path, mask_path, save_path, simple_size)
        
def split_label(label_csv, montage_root):
    df = pd.read_csv(label_csv)
    pos_list = df.loc[df['gt'] == 1]['id'].tolist()
    pos_list = [str(i) for i in pos_list]
    img_list = os.listdir(montage_root)
    for i in range(len(img_list)):
        if img_list[i].split('t')[0] in pos_list:
            os.system('mv ' + montage_root + '/' + img_list[i] + ' ' + montage_root + '/pos_subj')
            print (img_list[i])
    
#single_multi('/nfs/masi/gaor2/data/NLST/multiS_new/img/203791time2000.nii.gz', '/nfs/masi/gaor2/data/NLST/multiS_new/mask/203791time2000.nii.gz','/nfs/masi/gaor2/data/NLST/multiS_new/montage/203791time2000.jpeg', 224)

#folder_multi('/nfs/masi/NLST/registration/20191128_NLST_reg_affine_ori_mask_local/After_REG/multiS_new3/img', 224)

reg_folder_multi('/nfs/masi/NLST/registration/20191128_NLST_reg_affine_ori_mask_local/After_REG/multiS_new4/img', [254, 380])

#split_label(label_csv = '/nfs/masi/MCL/file/clinical/new_mcl_label.csv', montage_root = '/nfs/masi/gaor2/data/MCL/2D_norm/montage')
#single_folder(img_root = '/nfs/masi/gaor2/tmp/train_nifti/guass', save_root = '/nfs/masi/gaor2/tmp/train_nifti/guass_mont', img_size = 224)

