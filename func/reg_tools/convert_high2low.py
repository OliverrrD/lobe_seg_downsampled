import nibabel as nib

import nibabel as nib
import numpy as np
import os
import pandas as pd
from skimage import transform

import argparse



def crop_img_roi(ori_path, mask_path, xb, xe, yb, ye, zb, ze, save_npy=None, save_nii=None):
    data_nii = nib.load(ori_path)
    mask_nii = nib.load(mask_path)
    data = data_nii.get_data()
    mask = mask_nii.get_data()
    print (mask.shape, data.shape)
    mask[mask > 0.1] = 1.
    mask[mask <= 0.1] = 0.
    mask[mask != mask] = 0.
#    data[data!=data] = 0

    std_data = data[xb:xe, yb:ye, zb:ze]
    mask = mask[xb:xe, yb:ye, zb:ze]

    std_copy = std_data.copy()
    std_data[std_copy < -1000] = -1000
    std_data[std_copy < -1200] = -1200
    std_data[std_copy > 600] = 600
    # std_data = (std_data - std_data.min()) / (std_data.max() - std_data.min()) * 255
    std_data = (std_data + 1200) / 1800 * 255  # change at 1229
    # + 170. * (1 - mask)  # .astype('uint8')
    std_data = std_data * mask + 170 * (1 - mask)
    bone_thresh = 210
    std_data[std_data > bone_thresh] = 170
   # std_data[std_copy == 0] = 170
    std_data[std_copy != std_copy] = 170
    #std_data = std_data.astype("uint8")

    save_data = np.swapaxes(std_data, 0, 2)
    save_data = np.flip(save_data, axis=1)

    new_size = [325, 254, 380]   #

    save_data = transform.resize(save_data, new_size, mode='edge', preserve_range='True')
    save_data = save_data.astype('uint8')
    if save_nii != None:
        new_affine = np.array([[0, 0, -1, 0], [0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]) # for 1x1x1
        # new_affine = np.array([[0, 0, -0.8, 0], [0, -0.883, 0, 0], [0.883, 0, 0, 0], [0, 0, 0, 1]])
        new_nii = nib.Nifti1Image(save_data, new_affine)
        nib.save(new_nii, save_nii)
    if save_npy != None:
        new_data = np.expand_dims(save_data, axis=0)
        np.save(save_npy, new_data)

    # for size 620, 620, 600 , the roi size should be 430, 288, 406  # nii x, y, z
    # so, the index xb, xe, yb, ye, zb, ze is: [(620 - 430) / 2: 430 + (620 - 430) / 2, (620 - 288) / 2: 288 + (620 - 288) / 2, (600 - 406) / 2: 318 + (600 - 406) / 2 ]
    # which is: [95: 525, 166: 454, 97: 503 ]


    # for 540, 540, 480 is --xb 80 --xe 460 --yb 145 --ye 399 --zb 141 --ze 435,   the size is 380 254, 325 # nii x, y, z

def debug():
    crop_img_roi('/nfs/masi/xuk9/NLST/registration/affine_niftyreg/20200201_NLST_missing_data_local_config_v4/interp/ori/134532time2001.nii.gz',
                 '/nfs/masi/xuk9/NLST/registration/affine_niftyreg/20200201_NLST_missing_data_local_config_v4/interp/mask/134532time2001.nii.gz',
                 95, 525, 166, 454, 97, 503,
                  save_nii = '/nfs/masi/NLST/registration/20200201_NLST_missing_data_local_config_v4/After_REG/reg_crop/134532time2001.nii.gz')


if __name__ == "__main__":
    debug()
    # parser = argparse.ArgumentParser(description='crop the image with lung mask')
    # parser.add_argument('--ori_path', type=str)
    # parser.add_argument('--mask_path', type=str)
    # parser.add_argument('--save_npy', type=str)
    # parser.add_argument('--save_nii', type=str)
    # parser.add_argument('--crop_index', type=list, help='the index of xb, xe, yb, ye, zb, ze')
    # parser.add_argument('--xb', type=int)
    # parser.add_argument('--xe', type=int)
    # parser.add_argument('--yb', type=int)
    # parser.add_argument('--ye', type=int)
    # parser.add_argument('--zb', type=int)
    # parser.add_argument('--ze', type=int)
    # args = parser.parse_args()
    # crop_img_roi(args.ori_path, args.mask_path, args.xb, args.xe,
    #              args.yb, args.ye, args.zb,args.ze, args.save_npy, args.save_nii)




