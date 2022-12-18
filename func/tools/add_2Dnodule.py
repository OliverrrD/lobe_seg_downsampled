from skimage.draw import polygon, ellipse_perimeter
from skimage.morphology import * 
import nibabel as nb
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import numpy as np
import pandas as pd

def add_nodule(img, center, r, noise, orientation, k_dil, k_ero, coff, var ):
    rr, cc = ellipse_perimeter(center[0], center[1], r[0], r[1], orientation)
    rr = rr + np.random.normal(0, noise[0], len(rr))
    cc = cc + np.random.normal(0, noise[1], len(cc))
    x_list, y_list = polygon(rr, cc)
    no_img = np.zeros(img.shape, dtype = np.uint8)
    no_img[x_list, y_list] = 255
    img_dil = dilation(no_img, disk(k_dil))
    img_ero = erosion(img_dil, disk(k_ero))
    coors = np.mgrid[:img.shape[0], :img.shape[1]]
    distance_map = (coors[0] - center[0]) ** 2 + ((coors[1] - center[1])) ** 2 #  / r[1] * r[0]
    heat_map = coff* np.exp(-distance_map / var)
    mask = img_ero * heat_map
    new_img = img + mask
    new_mask = 255 * np.exp(-distance_map / 1000.0)
    return new_mask, new_img

def pos_csv():
    r1 = np.random.normal(7, 2)
    r2 = r1 + np.random.normal(0, 1)
    r1, r2 = int(abs(r1)), int(abs(r2))
    n1 = np.random.normal(1.5,0.5)
    n2 = n1 + np.random.normal(0, 0.5)
    n1, n2 = abs(n1), abs(n2)
    orientation = np.random.normal(0,1)
    k_dil = np.random.normal(1.5, 1)
    k_ero= np.random.normal(1.5, 1)
    k_dil = 1 + abs(k_dil - 1)
    k_ero = 1 + abs(k_ero - 1)
    coff = np.random.normal(0.8, 0.2)
    coff = max(coff, 0.2)
    coff = min(1, coff)
    var = np.random.normal(60, 30)
    var = max(20, var)
    return [r1,r2], ['%.2f'%n1,'%.2f'%n2],'%.2f'%orientation,'%.2f'%k_dil, '%.2f'%k_ero, '%.2f'%coff, '%.2f'%var, True

def neg_csv():
    r1 = np.random.normal(4, 1)
    r2 = r1 + np.random.normal(0, 0.8)
    r1, r2 = int(max(r1, 1)), int(max(r2,1))
    n1 = np.random.normal(0.8,0.5)
    n2 = n1 + np.random.normal(0, 0.2)
    n1, n2 = abs(n1), abs(n2)
    orientation = np.random.normal(0,1)
    k_dil = np.random.normal(1, 0.5)
    k_ero= np.random.normal(1, 0.5)
    k_dil = 1 + abs(k_dil - 1)
    k_ero = 1 + abs(k_ero - 1)
    coff = np.random.uniform(0.6, 0.2)
    coff = max(coff, 0.2)
    coff = min(1, coff)
    var = np.random.normal(50, 20)
    var = max(10, var)
    return [r1,r2], ['%.2f'%n1,'%.2f'%n2],'%.2f'%orientation,'%.2f'%k_dil, '%.2f'%k_ero, '%.2f'%coff, '%.2f'%var, False

#def nodule2scan(npy, info, center, r, noise, orientation, k_dil, k_ero, coff, var):
def get_slice_center(npy):
    num_slice = npy.shape[0]
    slice_list = []
    center_list = []
    #print (num_slice, 'num_slice')
    while True:
        flag = 0
        index = np.random.randint(10, num_slice - 10)
        thresh = threshold_otsu(npy[index])
        binary = npy[index] <= thresh
        for i in range(5):
            c0 = np.random.randint(5, npy[index].shape[0] -5)
            c1 =np.random.randint(5, npy[index].shape[1] - 5)
            if binary[c0, c1] == 1:
                flag = 1
                break
        if flag == 1:
            slice_list.append(index)
            center_list.append([c0, c1])
        if len(slice_list) >= 5:
            break
    return slice_list, center_list

df = pd.read_csv('/nfs/masi/NLST/file/csv_file/label_sumulation.csv')
df = df.loc[df['gt'] == 0][:10000]
data_root = '/nfs/masi/NLST/DSB_File/no_diag/prep/prep_noreport_link'
all_slice, all_center = [], []
for i, item in df.iterrows():
    if (i % 100 == 0): print (i, len(df))
    try:
        npy = np.load(data_root + '/' + item['sess'] + '_clean.npy')[0]
        slice_list, center_list = get_slice_center(npy)
        all_slice.append(slice_list)
        all_center.append(center_list)
    except: 
        all_slice.append('')
        all_center.append("")
df['slice'] = all_slice
df['center'] = all_center
df.to_csv('/nfs/masi/NLST/file/csv_file/label_sumulation_neg.csv', index = False)