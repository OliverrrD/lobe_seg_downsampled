import glob
import os

from matplotlib import pyplot as plt

import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    Spacingd,
    AddChanneld,
    Flipd,
    Resized,
    Orientationd,
    ScaleIntensityRanged,
    EnsureTyped,
    Identityd,
ThresholdIntensityd,
)
import math

from tqdm import tqdm

from dataloader import MatchSized



HU_WINDOW = (-1024, 600)
if __name__ == "__main__":
    data_dir='/home/dongc1/data/vlsp/train_256/'
    label_dir='/home/dongc1/data/vlsp/label_256/'
    images = glob.glob(os.path.join(data_dir, "*.npy"))
    labels = glob.glob(os.path.join(label_dir, "*.npy"))
    # args = sys.argv[1:]
    # main(*args)
    PIX_DIM=[2,2,2]
    # data={"image":"/home/dongc1/data/TotalSegmentor/npy/train_512/s0556.npy",
    #       "label":"/home/dongc1/data/TotalSegmentor/npy/label_512/s0556.npy"}
    data_dict = [
        {"image": image, "label":label}
        for image,label in zip(images,labels)
    ]
    transforms_TS = Compose([
        LoadImaged(keys=["image", "label"]),
        # Spacingd(keys=["image", "label"], pixdim=PIX_DIM, mode=("bilinear", "nearest")),
        ThresholdIntensityd(keys=["label"], threshold=0.5, above=False, cval=1.0),
        EnsureTyped(keys=["image", "label"], data_type="numpy"),
    ])
    for i, data in tqdm(enumerate(data_dict)):
        npy = transforms_TS(data)
        plt.imshow(npy["image"][0, :, :, 130], cmap='gray')
        print(npy["image"].shape)
        print(npy["label"].shape)
        plt.imshow(npy["label"][0,:,:,130], cmap='viridis', alpha=0.7)
        # # plt.imshow(img_array1[0, :, :, 100], cmap='gray')
        # # print(img_array1.shape)
        # # print(img_array1.shape)
        # # plt.imshow(img_array2[0, :, :, 100], cmap='gray', alpha=0.7)
        plt.show()
