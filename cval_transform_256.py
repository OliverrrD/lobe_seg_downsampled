"""
Apply transformations to dataset before model training for a faster dataloader
NOTE:
- affine matrix of TotalSegmentator and VUMC are coded differently.
- i.e. same orientation transform will result in different image space
- solution: manually correct after orientation transform with flip operation
"""

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
    Identityd
)
import glob
import os
import sys
import numpy as np
from tqdm import tqdm
from dataloader import MatchSized

PIX_DIM = (1,1,1)
HU_WINDOW = (-1024, 600)

transforms_TS = Compose([
    LoadImaged(keys=["image", "label"]),
    AddChanneld(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=PIX_DIM, mode=("bilinear", "nearest")),
    Resized(keys=["image", "label"], spatial_size=(256,256,256), mode=("trilinear", "nearest")),
    MatchSized(keys=["image", "label"], mode="crop"),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Flipd(keys=["image", "label"], spatial_axis=0),
    ScaleIntensityRanged(keys=["image"], a_min=HU_WINDOW[0], a_max=HU_WINDOW[1], b_min=0.0, b_max=1.0,
                        clip=True),
    EnsureTyped(keys=["image", "label"], data_type="numpy"),
])

transforms_vlsp = Compose([
    LoadImaged(keys=["image", "label"]),
    AddChanneld(keys=["image", "label"]),
    #Spacingd(keys=["image", "label"], pixdim=PIX_DIM, mode=("bilinear", "nearest")),
    Resized(keys=["image", "label"], spatial_size=(256,256,256), mode=("trilinear", "nearest")),
    MatchSized(keys=["image", "label"], mode="crop"),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["image"], a_min=HU_WINDOW[0], a_max=HU_WINDOW[1], b_min=0.0, b_max=1.0,
                        clip=True),
    EnsureTyped(keys=["image", "label"], data_type="numpy"),
])

def main(data_dir, label_dir, data_out_dir, label_out_dir, transforms=transforms_vlsp):
    train_images = glob.glob(os.path.join(data_dir, "*.nii.gz"))
    train_file_names = []
    for i in train_images:
        name, suffix = os.path.splitext(os.path.basename(i))
        if suffix == ".mhd":
            train_file_names.append(f"{name}_LobeSegmentation.nrrd")
        elif suffix == ".gz":
            fname = f"{name[:-4]}_LobeSegmentation.nii.gz" if name[1] == '.' else f"lvlsetseg_{name[:-4]}.nii.gz"
            train_file_names.append(fname)
    train_labels = [os.path.join(label_dir, name) for name in train_file_names]

    data_dict = [
        {"image":image, "label":label}
        for image, label in zip(train_images, train_labels)
    ]
    npy_names = [f"{os.path.basename(img).split('.nii.gz')[0]}.npy" for img in train_images]

    # apply transforms
    for i, data in tqdm(enumerate(data_dict)):
        npy = transforms(data)
        with open(os.path.join(data_out_dir, npy_names[i]), 'wb') as f:
            np.save(f, npy["image"])
        with open(os.path.join(label_out_dir, npy_names[i]), 'wb') as f:
            np.save(f, npy["label"])

if __name__ == "__main__":
    args = sys.argv[1:]
    main(*args)