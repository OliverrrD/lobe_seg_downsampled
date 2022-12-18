import glob
import os
import sys
from tqdm import tqdm
import nibabel as nib
import matplotlib.pyplot as plt


def main(data_dir):
    images = glob.glob(os.path.join(data_dir, "*.nii.gz"))
    data_dict = [
        {"image": image}
        for image in images
    ]
    headerList = []
    total=0
    # apply transforms
    for i, data in tqdm(enumerate(data_dict)):
        image=nib.load(data["image"])
        header=image.header
        headerList.append(header['pixdim'][1:4])
        if header['pixdim'][1:4][0]-1.5<0.001 and \
            header['pixdim'][1:4][1]-1.5<0.001 and \
            header['pixdim'][1:4][2]-1.5<0.001:
            total+=1
    plt.plot(range(len(headerList)), headerList)
    print(headerList)
    print(total)
    plt.show()


if __name__ == "__main__":
    args = sys.argv[1:]
    main(*args)