from skimage import transform
import nibabel as nib
import argparse
import os
import numpy as np

img_b = nib.load('/share2/gaor2/tmp/reg/76621363time20130228.nii.gz')
header = img_b.header

img = img_b.get_data()

dim = (340, 230, 310)

x_b = ( dim[0] - header['dim'][1] ) // 2
y_b = ( dim[1] - header['dim'][2]) // 2
z_b = ( dim[2] - header['dim'][3]) // 2

x_e = x_b + header['dim'][1]
y_e = y_b + header['dim'][2]
z_e = z_b + header['dim'][3]

img_pad = 170 * np.ones(dim, dtype = np.uint8)

print (x_e - x_b, y_e - y_b, z_e - z_b)

print (img.shape)

print (x_b, x_e, '--', y_b, y_e)

img_pad[x_b:x_e, y_b:y_e, z_b:z_e] = img

img_nib = nib.Nifti1Image(img_pad, np.eye(4))

nib.save(img_nib, '/share2/gaor2/tmp/reg/76621363time20130228_pad.nii.gz')

