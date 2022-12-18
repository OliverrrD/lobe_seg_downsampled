from skimage import transform
import nibabel as nib
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Resample The NIFTI image')
    parser.add_argument('--ori', type=str,
                        help='The original image path you want to resample')
    parser.add_argument('--out', type=str, 
                        help='The output path of the generated image')
    parser.add_argument('--dim', type=list, default = [380, 380, 350], #[416, 416, 400],   # [400, 400, 375],     # 432, 432, 356
                        help='the new image dim')
    args = parser.parse_args()
    img = nib.load(args.ori)
    header = img.header
    print (header['dim'][1].dtype, args.dim[0])
    x_b = (header['dim'][1] - args.dim[0]) // 2
    y_b = (header['dim'][2] - args.dim[1]) // 2
    z_b = (header['dim'][3] - args.dim[2]) // 2
    
    os.system('fslroi ' + args.ori + ' ' + args.out + ' ' + str(x_b) + ' ' + 
             str(args.dim[0]) + ' ' + str(y_b) + ' ' + str(args.dim[1]) + ' ' + str(str(z_b)) + ' ' + str(args.dim[2]))
    
if __name__ == '__main__':
    main()
