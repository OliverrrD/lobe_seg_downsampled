import nibabel as nib
import numpy as np
import os
import pandas as pd


def get_nomask_reg():
    ori_path = '/nfs/masi/xuk9/NLST/registration/affine_niftyreg/20191128_NLST_reg_affine_ori_mask_local/interp/ori/217438time1999.nii.gz'
    data_nii = nib.load(ori_path)
    new_data = data_nii.get_data()
    std_data = new_data[99: 529, 155: 435, 99: 529]
    save_data = np.swapaxes(std_data, 0, 2)
    save_data = np.flip(save_data, axis = 1)
    print (save_data.shape)
    new_data = np.expand_dims(save_data, axis = 0)
    np.save('/nfs/masi/gaor2/data/NLST/DSB_File/affreg/prep_nomask/217438time1999_clean.npy', new_data)
    new_affine = np.array([[0, 0, -1, 0], [0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    new_nii = nib.Nifti1Image(save_data, new_affine)
    nib.save(new_nii, '/nfs/masi/gaor2/data/NLST/reg_nomask/217438time1999/217438time1999.nii.gz')

# ------------- step 1 ------------------# 
# do the statistic for the lung mask 
def get_mask_roi(mask_path):
    img_nii = nib.load(mask_path)
    img = img_nii.get_data()
    img[img < 0] = 0
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
    x_begin, x_end = x_list[0] - int(0.05 * len(x_list)), x_list[-1] + int (0.05 * len(x_list))
    y_begin, y_end = y_list[0] - int(0.05 * len(y_list)), y_list[-1] + int (0.05 * len(y_list))
    z_begin, z_end = z_list[0] , z_list[-1]
    return x_begin, x_end, y_begin, y_end, z_begin, z_end

def get_static_mask(mask_root):
    mask_list = os.listdir(mask_root)
    xb_list, xe_list, yb_list, ye_list,zb_list, ze_list = [],[],[],[],[],[]
    for i in range(len(mask_list)):
        print (i, len(mask_list))
        if i > 20: break
        x_begin, x_end, y_begin, y_end, z_begin, z_end = get_roi(mask_root + '/' + mask_list[i])
        print (x_begin, x_end, y_begin, y_end, z_begin, z_end)
        xb_list.append(x_begin)
        xe_list.append(x_end)
        yb_list.append(y_begin)
        ye_list.append(y_end)
        zb_list.append(z_begin)
        ze_list.append(z_end)
    return xb_list, xe_list, yb_list, ye_list,zb_list, ze_list

def crop_img_roi(ori_path, mask_path, xb, xe, yb, ye, zb, ze, save_npy = None, save_nii = None):
    data_nii = nib.load(ori_path)
    mask_nii = nib.load(mask_path)
    data = data_nii.get_data()
    mask = mask_nii.get_data()
    mask[mask > 0] = 1
    mask[mask <= 0] = 0
    new_data = data * mask
    std_data = new_data[xb:xe, yb:ye, zb:ze]
    
    std_copy = std_data.copy()
    std_data[std_data < -1200] = -1200
    std_data[std_data > 600] = 600
    #std_data = (std_data - std_data.min()) / (std_data.max() - std_data.min()) * 255
    std_data = (std_data + 1200) / 1800 * 255    # change at 1229
    bone_thresh = 210
    std_data[std_data > bone_thresh] = 170
    std_data[std_copy == 0] = 170
    std_data = std_data.astype("uint8")
    
    save_data = np.swapaxes(std_data, 0, 2)
    save_data = np.flip(save_data, axis = 1)
    if save_nii != None:
        new_affine = np.array([[0, 0, -1, 0], [0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
        new_nii = nib.Nifti1Image(save_data, new_affine)
        nib.save(new_nii, save_nii)
    if save_npy != None:
        new_data = np.expand_dims(save_data, axis = 0)
        np.save(save_npy, new_data)


if __name__ == "__main__":
    ori_root = '/nfs/masi/xuk9/NLST/registration/affine_niftyreg/20191128_NLST_reg_affine_ori_mask_local/interp/ori'
    mask_root = '/nfs/masi/xuk9/NLST/registration/affine_niftyreg/20191128_NLST_reg_affine_ori_mask_local/interp/mask'
    save_npy_root = '/nfs/masi/gaor2/data/NLST/DSB_File/affreg/prep'
    save_nii_root  = '/nfs/masi/gaor2/data/NLST/reg_crop'

    df = pd.read_csv('/nfs/masi/gaor2/data/Cotrain/nlst/nlst_pos1yr.csv')
    need_list = df.loc[df['gt'] == 1]['item'].tolist()
    ori_list = os.listdir(ori_root)
    mask_list = os.listdir(mask_root)
    
    for i in range(len(need_list)):
        print (i, len(need_list), need_list[i])
        if i > 25: break
        if need_list[i] not in mask_list: 
            print (need_list[i] + ' is not complete')
            continue
        tmp_id = need_list[i].replace('.nii.gz', '')
        ori_path = ori_root + '/' + tmp_id + '.nii.gz'
        mask_path = mask_root + '/' + tmp_id + '.nii.gz'
        save_npy = save_npy_root + '/' + tmp_id  + '_clean.npy'
        if not os.path.exists(save_nii_root + '/' + tmp_id):
            os.mkdir(save_nii_root + '/' + tmp_id)
        save_nii = save_nii_root + '/' + tmp_id  + '/' + tmp_id+ '.nii.gz'
        crop_img_roi(ori_path, mask_path, 99, 529, 155, 435, 99, 529, save_npy, save_nii)  # size 430, 280, 430

