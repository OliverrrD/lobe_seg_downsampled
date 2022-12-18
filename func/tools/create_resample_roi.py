import nibabel as nib
import numpy as np
from scipy.ndimage.interpolation import zoom
import pandas as pd
from skimage import transform
import re

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

def save_roi_single(data_path, mask_path, save_path):
    center = get_roi(mask_path)
    mask_nii = nib.load(mask_path)
    img_nii = nib.load(data_path)
    resol = [mask_nii.header['pixdim'][1], mask_nii.header['pixdim'][2], mask_nii.header['pixdim'][3]]
    lens = [def_size * def_reso / i for i in resol]
    x_begin, x_end = center[0] - lens[0] / 2, center[0] + lens[0] / 2
    y_begin, y_end = center[1] - lens[1] / 2, center[1] + lens[1] / 2
    z_begin, z_end = center[2] - lens[2] / 2, center[2] + lens[2] / 2
    img_new = img_nii.get_data()
    x_begin, x_end, y_begin, y_end, z_begin, z_end = max(0, int(x_begin)), int(x_end), max(0, int(y_begin)), int(
        y_end), max(0, int(z_begin)), int(z_end)
    img_new = img_new[x_begin: x_end, y_begin: y_end, z_begin: z_end]
    img_save = transform.resize(img_new, [def_size, def_size, def_size], mode='edge', preserve_range='True')
    norm_roi(img_save, save_path)
#    save_nii = nib.Nifti1Image(img_save, np.eye(4))
#    nib.save(save_nii, save_path)

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
    nib.save(save_nii, save_path)
    

def norm_roi(img, save_path):
    img[img < -1000] = -1000
    img[img > 1000] = 1000
    img_new = img.astype(np.int16)
    save_nii = nib.Nifti1Image(img_new, np.eye(4))
    nib.save(save_nii, save_path)


data_root = '/nfs/masi/SPORE/nifti/combine'
mask_root = '/nfs/masi/xuk9/SPORE/registration/thorax_reg_roi_full/lung_mask'
save_root = '/nfs/masi/gaor2/data/SPORE/roi_norm'

mcl_csv = '/nfs/masi/gaor2/data/Cotrain/new_twoset/spore.csv'

def_size = 200
def_reso = 1.5

df = pd.read_csv(mcl_csv)

for i, item in df.iterrows():
    print (i, item['item'])

    subj = str(item['subject'])
    sess = str(item['session'])
    img = item['item']
    name_list = re.split('[time]', img.replace('.nii.gz', ''))
    #print (name_list)
    subj, sess = name_list[0], name_list[-1]
    try:
        save_roi_single(data_root + '/' + subj + '/' + sess + '/' + img, mask_root + '/' + img, save_root + '/' + img)
        # center = get_roi(mask_root + '/' + img)
        # mask_nii = nib.load(mask_root + '/' + img)
        # img_nii = nib.load(data_root + '/' + subj + '/' + sess + '/' +  img)
        # resol = [mask_nii.header['pixdim'][1], mask_nii.header['pixdim'][2], mask_nii.header['pixdim'][3]]
        # lens = [def_size * def_reso / i for i in resol]
        # x_begin, x_end = center[0] - lens[0] / 2, center[0] + lens[0] / 2
        # y_begin, y_end = center[1] - lens[1] / 2, center[1] + lens[1] / 2
        # z_begin, z_end = center[2] - lens[2] / 2, center[2] + lens[2] / 2
        # img_new = img_nii.get_data()
        # x_begin, x_end, y_begin, y_end, z_begin, z_end = max(0, int(x_begin)), int(x_end), max(0, int(y_begin)), int(
        #     y_end), max(0, int(z_begin)), int(z_end)
        # img_new = img_new[x_begin: x_end, y_begin: y_end, z_begin: z_end]
        #
        # img_save = transform.resize(img_new, [def_size, def_size, def_size], mode='edge', preserve_range='True')
        #
        # save_nii = nib.Nifti1Image(img_save, np.eye(4))
        # nib.save(save_nii, save_root + '/' + img)
    except:
        print ('wrong in ' + item['item'])



