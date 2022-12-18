import nibabel as nib
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori',help='The original data path that want to segment')
    parser.add_argument('--mask', help = 'The mask from segmentation')
    parser.add_argument('--out',help='out path of the segmented image')
    args = parser.parse_args()
    mask = nib.load(args.mask)
    img = nib.load(args.ori)
    mask_3d = mask.get_data()
    mask_3d = np.array([mask_3d >= 1][0]) 
    img_3d = img.get_data()
    print (img_3d.shape, mask_3d.shape)
    img_new = img_3d * mask_3d
    out = nib.Nifti1Image(img_new, img.affine, img.header)
    print (args.out)
    nib.save(out, args.out)
    
if __name__ == '__main__':
    main()