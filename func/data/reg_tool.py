import os
import pandas as pd
import nibabel as nib
import numpy as np
from skimage import transform, util
import re
import matplotlib.pyplot as plt
import SimpleITK as sitk
import nibabel as nib
from glob import glob


def get_norm_subsess(data_root, save_csv):
    dict_subsess = {}
    sess_list = os.listdir(data_root)
    for i in range(len(sess_list)):
        print (sess_list[i])
        sess_vec = re.split('[item]', sess_list[i] )
        sess_vec = [i for i in sess_vec if len(i) != 0]
        subj = sess_vec[0]
        sess = sess_vec[1]
        if subj not in dict_subsess.keys():
            dict_subsess[subj] = [sess]
        else:
            dict_subsess[subj].append(sess)
    sub_list, subsess_list = [], []
    for subj in dict_subsess.keys():
        sub_list.append(subj)
        subsess_list.append(sorted(dict_subsess[subj]))
    df = pd.DataFrame()
    df['subject'] = sub_list
    df['dates'] = subsess_list
    df.to_csv(save_csv)
    
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_slurm_scripts(csv_path, slurm_root, log_root, save_data_root, tmp_data_root, make_dir = True):
    df = pd.read_csv(csv_path)
    for i, item in df.iterrows():
            if i % 20 == 0: print (i)
            slm_name = slurm_root + '/' + str(item['subject']) + 'from' + str(item['move_item']) + 'to' +  str(item['ref_item']) + '.sh'
            f = open(slm_name, 'w')
            file_name = 'reg' + str(item['subject']) + 'time' + str(item['move_item'])
            save_path = os.path.join(save_data_root, file_name)
            tmp_path = os.path.join(tmp_data_root, file_name.replace('reg', 'tmp'))
            if make_dir:
                mkdir(save_path)
                mkdir(tmp_path)
            f.write('#!/bin/bash\n')
            f.write('#SBATCH --job-name=' + str(item['subject']) + 'from' + str(item['move_item']) + 'to' +  str(item['ref_item']) + '\n')
            f.write('#SBATCH --output=' + log_root + '/' + str(item['subject']) + 'from' + str(item['move_item']) + 'to' +  str(item['ref_item']) + '.txt\n')
            f.write('#SBATCH --ntasks=1\n')
            f.write('#SBATCH --time=07:00:00\n')
            f.write('#SBATCH --mem-per-cpu=11G\n')
            f.write('sh /home/local/VANDERBILT/gaor2/code/RNNLung/func/tools/REGLUNG.sh ' + item['ref_path'] + ' ' 
                    + item['move_path'] + ' ' + save_path + ' ' + file_name + ' ' + tmp_path)
            f.write('\ndate')
            f.close()

def get_reg_pair(data_root, save_csv_path):
    subj_list = os.listdir(data_root)
    move_item_list = []
    ref_item_list = []
    move_path_list = []
    ref_path_list = []
    save_subj_list = []
    for i in range(len(subj_list)):
        subj_path = os.path.join(data_root, subj_list[i])
        sess_list = os.listdir(subj_path)
        if len(sess_list) > 1:
            max_index = sess_list.index(max(sess_list))
            for j in range(len(sess_list)):
                if j == max_index:
                    continue
                move_path = subj_path + '/' + sess_list[j]
                ref_path = subj_path + '/' + sess_list[max_index]
                move_item = os.listdir(move_path)[0]
                ref_item = os.listdir(ref_path)[0]
                move_item_list.append(str(sess_list[j]))
                ref_item_list.append(str(sess_list[max_index]))
                move_path_list.append(move_path + '/' + move_item)
                ref_path_list.append(ref_path + '/' + ref_item)
                save_subj_list.append(str(subj_list[i]))
                
    data = pd.DataFrame()
    print (len(save_subj_list) , len(move_item_list))
    data['subject'] = save_subj_list
    data['move_item'] = move_item_list
    data['ref_item'] = ref_item_list
    data['move_path'] = move_path_list
    data['ref_path'] = ref_path_list
    data.to_csv(save_csv_path)
    
def get_reg_pair_spread(data_root, save_csv_path):
    sess_list = os.listdir(data_root)
    move_item_list = []
    ref_item_list = []
    move_path_list = []
    ref_path_list = []
    save_subj_list = []
    subj_sess = {}
    for i in range(len(sess_list)):
        tmp_vec = re.split('[time]', sess_list[i])
        subj = tmp_vec[0]
        if subj not in subj_sess.keys():
            subj_sess[subj] = []
        subj_sess[subj].append(sess_list[i])
    for key in subj_sess.keys():
        if len(subj_sess[key]) <= 1:
            continue
        else:
            tmp_vec = sorted(subj_sess[key])
            for i in range(len(tmp_vec) - 1):
                move_item_list.append(tmp_vec[i][-8:])
                ref_item_list.append(tmp_vec[-1][-8:])
                move_path_list.append(data_root + '/' + tmp_vec[i] + '/' + tmp_vec[i] + '.nii.gz')
                ref_path_list.append(data_root + '/' + tmp_vec[-1] + '/' + tmp_vec[-1] + '.nii.gz')
                save_subj_list.append(str(key))
                
    data = pd.DataFrame()
    print (len(save_subj_list) , len(move_item_list))
    data['subject'] = save_subj_list
    data['move_item'] = move_item_list
    data['ref_item'] = ref_item_list
    data['move_path'] = move_path_list
    data['ref_path'] = ref_path_list
    data.to_csv(save_csv_path)    
    
def run_all_scripts(scripts_root, reg_root, min_index, max_index, use_exists = True):
    scripts_list = os.listdir(scripts_root)
#    f = open('/share3/gaor2/share5backup/data/MCL/Registration/slurm/existed.txt', 'w')
    for i in range(min_index, max_index):
        #if (i < 800): continue
        
        if use_exists:
            name_vec = re.split('[fritemo.]', scripts_list[i])
            new_vec = [i for i in name_vec if len(i) != 0]
            subj_name = new_vec[0]
            move_sess = new_vec[1]
       
            if os.path.exists(reg_root + '/reg' + subj_name + 'time' + move_sess) and \
            len(os.listdir(reg_root + '/reg' + subj_name + 'time' + move_sess)) != 0:
#                f.write(reg_root + '/reg' + subj_name + 'time' + move_sess + '\n')
                print (reg_root + '/reg' + subj_name + 'time' + move_sess + ' is existed!')
                continue
#        print (reg_root + '/reg' + subj_name + 'time' + move_sess + ' is existed!')
        print ('sbatch ' + scripts_root + '/' + scripts_list[i])
        os.system('sbatch ' + scripts_root + '/' + scripts_list[i])
#    f.close()
    
def get_NLST_pair(data_root, save_csv_path):
    subj_list = os.listdir(data_root)
    filt = '*.nii.gz'
    move_item_list = []
    ref_item_list = []
    move_path_list = []
    ref_path_list = []
    save_subj_list = []
    
    
    def find_latest_path(path_list):
        date_list = []
        for i in range(len(path_list)):
            sess_name = path_list[i].split('/')[-1]
            date = float(sess_name[6:10] + sess_name[0:2] + sess_name[3:5])
            date_list.append(date)
        latest_index = date_list.index(max(date_list))
        return latest_index
    
    def find_max_item(item_list):
        size_list = []
        for i in range(len(item_list)):
            item = item_list[i]
            size_list.append(os.path.getsize(item))
        max_index = size_list.index(max(size_list))
        return max_index
            
    for i in range(len(subj_list)):
        print (i, subj_list[i])
        subj_path = os.path.join(data_root, subj_list[i])
        sess_list = os.listdir(subj_path)
        #sess_list = glob(os.path.join(subj_path, filt))
        if len(sess_list) == 1:
            continue
        latest_index = find_latest_path(sess_list)
        
        latest_sess = sess_list[latest_index]
        latest_item_list = glob(os.path.join(subj_path + '/' + latest_sess, filt))
        max_index = find_max_item(latest_item_list)
        max_item_path = latest_item_list[max_index]
        
        for j in range(len(sess_list)):
                if j == latest_index:
                    continue
                
                item_list = glob(os.path.join(subj_path + '/' + sess_list[j], filt))
#                 if len(item_list) == 0:
#                     print ('no nifti in ', item_list)
#                     continue
                tmp_max_index = find_max_item(item_list)
                #move_path = item_list[tmp_max_index]
                #ref_path = max_item_path
                
                move_item_list.append(sess_list[j])
                print (sess_list)
                print (max_index)
                ref_item_list.append(sess_list[latest_index])
                move_path_list.append(item_list[tmp_max_index])
                ref_path_list.append(max_item_path)
                save_subj_list.append(subj_list[i])
         
    data = pd.DataFrame()
    print (len(save_subj_list) , len(move_item_list))
    data['subject'] = save_subj_list
    data['move_item'] = move_item_list
    data['ref_item'] = ref_item_list
    data['move_path'] = move_path_list
    data['ref_path'] = ref_path_list
    data.to_csv(save_csv_path)

            
def get_slurm_scripts_NLST(csv_path, slurm_root, log_root, save_data_root, tmp_data_root, make_dir = True):
    df = pd.read_csv(csv_path)
    for i, item in df.iterrows():
            print (i, item['subject'])
            slm_name = slurm_root + '/' + str(item['subject']) + 'from' + str(item['move_item'][6:10]) + 'to' +  str(item['ref_item'][6:10]) + '.sh'
            f = open(slm_name, 'w')
            file_name = 'reg' + str(item['subject']) + 'time' + str(item['move_item'][6:10])
            save_path = os.path.join(save_data_root, file_name)
            tmp_path = os.path.join(tmp_data_root, file_name.replace('reg', 'tmp'))
            if make_dir:
                mkdir(save_path)
                mkdir(tmp_path)
            f.write('#!/bin/bash\n')
            f.write('#SBATCH --job-name=' + str(item['subject']) + 'from' + str(item['move_item'][6:10]) + 'to' +  str(item['ref_item'][6:10]) + '\n')
            f.write('#SBATCH --output=' + log_root + '/' + str(item['subject']) + 'from' + str(item['move_item'][6:10]) + 'to' +  str(item['ref_item'][6:10]) + '.txt\n')
            f.write('#SBATCH --ntasks=1\n')
            f.write('#SBATCH --time=07:00:00\n')
            f.write('#SBATCH --mem-per-cpu=11G\n')
            f.write('sh /home/local/VANDERBILT/gaor2/code/RNNLung/func/tools/REGLUNG.sh ' + item['ref_path'] + ' ' 
                    + item['move_path'] + ' ' + save_path + ' ' + file_name + ' ' + tmp_path)
            f.write('\ndate')
            f.close()        
        


def create_subj_dict():
    # for generate the second registration
    cancer_root = '/media/gaor2/46324cd9-d012-4189-b58d-54605a1828aa/NLST/NLST/cancer'
    cancer_subj = os.listdir(cancer_root)

    nocancer_root = '/media/gaor2/46324cd9-d012-4189-b58d-54605a1828aa/NLST/NLST/nocancer'
    nocancer_subj = os.listdir(nocancer_root)

    subj_dict = {}
    for subj in cancer_subj:
        tmp_sess = os.listdir(cancer_root + '/' + subj)
        tmp_sess = [i[6:10] for i in tmp_sess]
        subj_dict[subj] = tmp_sess

    for subj in nocancer_subj:
        tmp_sess = os.listdir(nocancer_root + '/' + subj)
        tmp_sess = [i[6:10] for i in tmp_sess]
        subj_dict[subj] = tmp_sess
    
def diff_noreg_reg():
    # find the different item between the noreg and reg record, for generate the second registration
    df_noreg = pd.read_csv('/share3/gaor2/share5backup/data/NLST/NLSTnorm/noreg/long_labels.csv')
    df_reg = pd.read_csv('/share3/gaor2/share5backup/data/NLST/NLSTnorm/reg/long_labels_noempty.csv')
    noreg_list = df_noreg['item'].tolist()
    reg_list = df_reg['item'].tolist()
    reg_list = [i[3:] for i in reg_list]
    new_list =  list(set(noreg_list) - set(reg_list))
    new_subj_set = set([i[:6] for i in new_list])
    print ((new_list))
    
def get_second_table():
    subj_list, move_item, ref_item = [], [], []
    for i in range(len(new_list)):
        subj = new_list[i][:6]
        subj_list.append(subj)
        move_item.append(new_list[i][10:14])
        ref_item.append(max(subj_dict[subj]))
    data = pd.DataFrame()
    data['subject'] = subj_list
    data['move_item'] = move_item
    data['ref_item'] = ref_item
    data.to_csv('/share2/gaor2/NLST/Registration/second_table.csv', index = False)
    
def copy_spread_from_table(csv_path, data_root, new_root):
    df = pd.read_csv(csv_path)
    copy_list = []
    for i, item in df.iterrows():
        print (str(item['subject']) + 'time' + str(item['move_item']))
        copy_list.append(str(item['subject']) + 'time' + str(item['move_item']))
        copy_list.append(str(item['subject']) + 'time' + str(item['ref_item']))
    copy_list = list(set(copy_list))
    for i in range(len(copy_list)):
        try:
            os.system('cp -r ' + data_root[1] + '/' + copy_list[i] + ' ' + new_root)
            print (i, len(copy_list), 'cp -r ' + data_root[1] + '/' + copy_list[i] + ' ' + new_root)
        except:
            os.system('cp -r ' + data_root[0] + '/' + copy_list[i] + ' ' + new_root + '/')
            print ('cp -r ' + data_root[0] + '/' + copy_list[i] + ' ' + new_root + '/')
        
def get_size_list(img_root):
    img_list = os.listdir(img_root)
    size_arr = np.zeros((len(img_list), 3))
    for i in range(len(img_list)):
        if i % 10 == 0: print (i)
        #if i > 100: break
        img_nib = nib.load(img_root + '/' + img_list[i])
        img_arr = img_nib.get_data()
        size_arr[i] = img_arr.shape
    return size_arr
        
        
        
        
        
        
        
        
        
        
        
        
        