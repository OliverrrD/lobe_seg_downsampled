from monai.transforms import (
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    AddChanneld,
    RandShiftIntensityd,
    RandAffined,
    ToTensord,
    EnsureTyped,
    SpatialPadd,
    SpatialPad, Resized, Flipd,
)
import nibabel as nib
from nilearn.image import resample_img
import pylab as plt
import nibabel as nb
import numpy as np

if __name__ == "__main__":
    data={"image":"/home/dongc1/data/00000582time20170403.nii.gz"}
    image=nib.load("/home/dongc1/data/00000582time20170403.nii.gz")
    print(image.affine)
    newAffine=np.zeros(image.affine.shape);
    for i in range(image.affine.shape[0]):
        for j in range(image.affine.shape[1]):
            if i==j and i==0:
                newAffine[i][j]=image.affine[0][0]*2
            elif i==j and i==1:
                newAffine[i][j] = -1*image.affine[0][0] * 2
            elif i==j and j==2:
                newAffine[i][j] = -1* image.affine[0][0] * 2
            else:
                newAffine[i][j] = image.affine[i][j]

    downsampled_nii = resample_img(image, target_affine=newAffine, interpolation='linear')
    print(downsampled_nii.shape)
    print("\n")
    print(downsampled_nii.affine)
    nib.save(downsampled_nii, '/home-nfs2/local/VANDERBILT/dongc1/downsample_582.nii.gz')