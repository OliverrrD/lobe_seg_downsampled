import argparse
from data_io import DataFolder, ScanWrapper
from utils import get_logger
from paral import AbstractParallelRoutine
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from utils import mkdir_p
import os
# from vis.pptx import save_pptx
import cv2 as cv


logger = get_logger('Clip with mask')


def _clip_image(image_data, clip_plane, num_clip=1, idx_clip=0):
    im_shape = image_data.shape

    # Get clip offset
    idx_dim = -1
    if clip_plane == 'sagittal':
        idx_dim = 0
    elif clip_plane == 'coronal':
        idx_dim = 1
    elif clip_plane == 'axial':
        idx_dim = 2
    else:
        raise NotImplementedError

    clip_step_size = int(float(im_shape[idx_dim]) / (num_clip + 1))
    offset = -int(float(im_shape[idx_dim]) / 2) + (idx_clip + 1) * clip_step_size

    clip_location = int(im_shape[idx_dim] / 2) - 1 + offset

    clip = None
    if clip_plane == 'sagittal':
        clip = image_data[clip_location, :, :]
        clip = np.flip(clip, 0)
        clip = np.rot90(clip)
    elif clip_plane == 'coronal':
        clip = image_data[:, clip_location, :]
        clip = np.rot90(clip)
    elif clip_plane == 'axial':
        clip = image_data[:, :, clip_location]
        clip = np.rot90(clip)
    else:
        raise NotImplementedError

    return clip

def _clip_image_sitk(image_data, clip_plane, num_clip=1, idx_clip=0):
    im_shape = image_data.shape

    # Get clip offset
    idx_dim = -1
    if clip_plane == 'sagittal':
        idx_dim = 2
    elif clip_plane == 'coronal':
        idx_dim = 1
    elif clip_plane == 'axial':
        idx_dim = 0
    else:
        raise NotImplementedError

    clip_step_size = int(float(im_shape[idx_dim]) / (num_clip + 1))
    offset = -int(float(im_shape[idx_dim]) / 2) + (idx_clip + 1) * clip_step_size

    clip_location = int(im_shape[idx_dim] / 2) - 1 + offset

    clip = None
    if clip_plane == 'sagittal':
        clip = image_data[:, :, clip_location]
        clip = np.flip(clip, 0)
    elif clip_plane == 'coronal':
        clip = image_data[:, clip_location, :]
        clip = np.flip(clip, 0)
    elif clip_plane == 'axial':
        clip = image_data[clip_location, :, :]
    else:
        raise NotImplementedError

    return clip

def _clip_image_RAS(image_data, clip_plane, num_clip=1, idx_clip=0):
    im_shape = image_data.shape

    # Get clip offset
    idx_dim = -1
    if clip_plane == 'sagittal':
        idx_dim = 0
    elif clip_plane == 'coronal':
        idx_dim = 1
    elif clip_plane == 'axial':
        idx_dim = 2
    else:
        raise NotImplementedError

    clip_step_size = int(float(im_shape[idx_dim]) / (num_clip + 1))
    offset = -int(float(im_shape[idx_dim]) / 2) + (idx_clip + 1) * clip_step_size

    clip_location = int(im_shape[idx_dim] / 2) - 1 + offset

    clip = None
    if clip_plane == 'sagittal':
        clip = image_data[-clip_location, :, :]
        clip = np.flip(clip, 0)
        clip = np.rot90(clip)
    elif clip_plane == 'coronal':
        clip = image_data[:, clip_location, :]
        clip = np.rot90(clip)
        clip = np.flip(clip, 1)
    elif clip_plane == 'axial':
        clip = image_data[:, :, clip_location]
        clip = np.rot90(clip)
        clip = np.flip(clip, 1)
    else:
        raise NotImplementedError

    return clip


def multiple_clip_overlay_with_mask(in_nii, in_mask, out_png, clip_plane='axial', img_vrange=(-1000,600), dim_x=4, dim_y=4):
    '''Creates overlay from input nifti file paths'''
    num_clip = dim_x * dim_y
    print(f'reading {in_nii}')
    print(f'reading {in_mask}')
    in_img = ScanWrapper(in_nii).get_data()
    in_mask_img = ScanWrapper(in_mask).get_data()

    pixdim = ScanWrapper(in_nii).get_header()['pixdim'][1:4]
    dim_physical = np.multiply(np.array(in_img.shape), pixdim).astype(int)

    clip_in_img_list = []
    clip_mask_img_list = []
    for idx_clip in range(num_clip):
        clip_in_img = _clip_image(in_img, clip_plane, num_clip, idx_clip)
        clip_mask_img = _clip_image(in_mask_img, clip_plane, num_clip, idx_clip)

        clip_in_img = cv.resize(clip_in_img, (dim_physical[0], dim_physical[1]), interpolation=cv.INTER_CUBIC)
        clip_mask_img = cv.resize(clip_mask_img, (dim_physical[0], dim_physical[1]),
                                  interpolation=cv.INTER_NEAREST)

        clip_in_img = np.concatenate([clip_in_img, clip_in_img], axis=1)
        clip_mask_img = np.concatenate(
            [np.zeros(clip_mask_img.shape, dtype=int),
             clip_mask_img], axis=1
        )
        clip_mask_img = clip_mask_img.astype(float)
        clip_mask_img[clip_mask_img == 0] = np.nan

        clip_in_img_list.append(clip_in_img)
        clip_mask_img_list.append(clip_mask_img)

    multiple_clip_overlay_with_mask_from_list(clip_in_img_list,
                                              clip_mask_img_list,
                                              out_png,
                                              img_vrange=img_vrange,
                                              mask_vrange=(np.min(in_mask_img), np.max(in_mask_img)),
                                              dim_x=4,
                                              dim_y=4)

def multiple_clip_overlay_with_mask_from_npy(in_img, in_mask, out_png, clip_plane='axial', img_vrange=(-1000,600), dim_x=4, dim_y=4):
    '''Creates overlay from input npy images of the same size'''
    num_clip = dim_x * dim_y
    clip_in_img_list = []
    clip_mask_img_list = []
    for idx_clip in range(num_clip):
        clip_in_img = _clip_image_RAS(in_img, clip_plane, num_clip, idx_clip)
        clip_mask_img = _clip_image_RAS(in_mask, clip_plane, num_clip, idx_clip)

        clip_in_img = np.concatenate([clip_in_img, clip_in_img], axis=1)
        clip_mask_img = np.concatenate(
            [np.zeros(clip_mask_img.shape, dtype=int),
             clip_mask_img], axis=1
        )
        clip_mask_img = clip_mask_img.astype(float)
        clip_mask_img[clip_mask_img == 0] = np.nan

        clip_in_img_list.append(clip_in_img)
        clip_mask_img_list.append(clip_mask_img)

    multiple_clip_overlay_with_mask_from_list(clip_in_img_list,
                                              clip_mask_img_list,
                                              out_png,
                                              img_vrange=img_vrange,
                                              mask_vrange=(np.min(in_mask), np.max(in_mask)),
                                              dim_x=4,
                                              dim_y=4)

def multiple_clip_overlay_with_mask_from_np_sitk(in_img, in_mask, out_png, clip_plane='axial', img_vrange=(-1000,600), dim_x=4, dim_y=4):
    '''Creates overlay from input npy images of the same size'''
    num_clip = dim_x * dim_y
    clip_in_img_list = []
    clip_mask_img_list = []
    for idx_clip in range(num_clip):
        clip_in_img = _clip_image_sitk(in_img, clip_plane, num_clip, idx_clip)
        clip_mask_img = _clip_image_sitk(in_mask, clip_plane, num_clip, idx_clip)

        clip_in_img = np.concatenate([clip_in_img, clip_in_img], axis=1)
        clip_mask_img = np.concatenate(
            [np.zeros(clip_mask_img.shape, dtype=int),
             clip_mask_img], axis=1
        )
        clip_mask_img = clip_mask_img.astype(float)
        clip_mask_img[clip_mask_img == 0] = np.nan

        clip_in_img_list.append(clip_in_img)
        clip_mask_img_list.append(clip_mask_img)

    multiple_clip_overlay_with_mask_from_list(clip_in_img_list,
                                              clip_mask_img_list,
                                              out_png,
                                              img_vrange=img_vrange,
                                              mask_vrange=(np.min(in_mask), np.max(in_mask)),
                                              dim_x=4,
                                              dim_y=4)

def multiple_clip_overlay_from_np_sitk(in_img, out_png, clip_plane='axial', img_vrange=(-1024,600), dim_x=4, dim_y=4):
    '''Creates overlay from input npy images of the same size'''
    num_clip = dim_x * dim_y
    clip_in_img_list = []
    for idx_clip in range(num_clip):
        clip_in_img = _clip_image_sitk(in_img, clip_plane, num_clip, idx_clip)
        clip_in_img = np.concatenate([clip_in_img, clip_in_img], axis=1)
        clip_in_img_list.append(clip_in_img)

    multiple_clip_overlay_from_list(clip_in_img_list,
                                              out_png,
                                              img_vrange=img_vrange,
                                              dim_x=4,
                                              dim_y=4)


def multiple_clip_overlay_from_list(clip_in_img_list, out_png, img_vrange=(-1024,600), dim_x=4, dim_y=4):
    '''Creates x by y clip of images from input list of clipped npy images. No mask overlay'''
    in_img_row_list = []
    for idx_row in range(dim_y):
        in_img_block_list = []
        for idx_column in range(dim_x):
            in_img_block_list.append(clip_in_img_list[idx_column + dim_x * idx_row])
        in_img_row = np.concatenate(in_img_block_list, axis=1)
        in_img_row_list.append(in_img_row)

    in_img_plot = np.concatenate(in_img_row_list, axis=0)

    fig, ax = plt.subplots()
    plt.axis('off')
    ax.imshow(
        in_img_plot,
        interpolation='bilinear',
        cmap='gray',
        norm=colors.Normalize(vmin=img_vrange[0], vmax=img_vrange[1]),
        alpha=0.8)

    print(f'Save to {out_png}')
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

def multiple_clip_overlay_with_mask_from_list(clip_in_img_list, clip_mask_img_list, out_png, img_vrange=(-1000,600), mask_vrange=(0,5), dim_x=4, dim_y=4):
    '''Creates overlay from input list of clipped npy images'''
    in_img_row_list = []
    mask_img_row_list = []
    for idx_row in range(dim_y):
        in_img_block_list = []
        mask_img_block_list = []
        for idx_column in range(dim_x):
            in_img_block_list.append(clip_in_img_list[idx_column + dim_x * idx_row])
            mask_img_block_list.append(clip_mask_img_list[idx_column + dim_x * idx_row])
        in_img_row = np.concatenate(in_img_block_list, axis=1)
        mask_img_row = np.concatenate(mask_img_block_list, axis=1)
        in_img_row_list.append(in_img_row)
        mask_img_row_list.append(mask_img_row)

    in_img_plot = np.concatenate(in_img_row_list, axis=0)
    mask_img_plot = np.concatenate(mask_img_row_list, axis=0)

    fig, ax = plt.subplots()
    plt.axis('off')
    ax.imshow(
        in_img_plot,
        interpolation='bilinear',
        cmap='gray',
        norm=colors.Normalize(vmin=img_vrange[0], vmax=img_vrange[1]),
        alpha=0.8)
    ax.imshow(
        mask_img_plot,
        interpolation='none',
        cmap='jet',
        norm=colors.Normalize(vmin=mask_vrange[0], vmax=mask_vrange[1]),
        alpha=0.5
    )

    print(f'Save to {out_png}')
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

def clip_overlay_with_mask(in_nii, in_mask, out_png):
    # Only do the clip on axial plane.
    print(f'reading {in_nii}')
    print(f'reading {in_mask}')
    in_img = ScanWrapper(in_nii).get_data()
    in_mask_img = ScanWrapper(in_mask).get_data()
    clip_in_img = in_img[:, :, int(in_img.shape[2] / 2.0)]
    clip_in_img = np.rot90(clip_in_img)
    clip_in_img = np.concatenate([clip_in_img, clip_in_img], axis=1)

    clip_mask_img = in_mask_img[:, :, int(in_img.shape[2] / 2.0)]
    clip_mask_img = np.rot90(clip_mask_img)
    clip_mask_img = np.concatenate(
        [np.zeros((in_img.shape[0], in_img.shape[1]), dtype=int),
         clip_mask_img], axis=1
    )
    clip_mask_img = clip_mask_img.astype(float)

    clip_mask_img[clip_mask_img == 0] = np.nan

    vmin_img = -1200
    vmax_img = 600

    fig, ax = plt.subplots()
    plt.axis('off')
    ax.imshow(
        clip_in_img,
        interpolation='bilinear',
        cmap='gray',
        norm=colors.Normalize(vmin=vmin_img, vmax=vmax_img),
        alpha=0.8)
    ax.imshow(
        clip_mask_img,
        interpolation='none',
        cmap='jet',
        norm=colors.Normalize(vmin=np.min(in_mask_img), vmax=np.max(in_mask_img)),
        alpha=0.5
    )

    print(f'Save to {out_png}')
    plt.savefig(out_png, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()


class ParalPPVMask(AbstractParallelRoutine):
    def __init__(self,
                 in_folder_obj,
                 out_mask_folder_obj,
                 out_clip_png_folder_obj,
                 num_process):
        super().__init__(in_folder_obj, num_process)
        self.out_mask_folder_obj = out_mask_folder_obj
        self.out_clip_png_folder_obj = out_clip_png_folder_obj

    def _run_single_scan(self, idx):
        # try:
        in_nii = self._in_data_folder.get_file_path(idx)
        out_mask_nii = self.out_mask_folder_obj.get_file_path(idx)
        out_png = self.out_clip_png_folder_obj.get_file_path(idx)

        if not os.path.exists(out_png):
            multiple_clip_overlay_with_mask(in_nii, out_mask_nii, out_png)
        else:
            logger.info(f'{out_png} already exists.')

        return out_png
        # except:
        #     print(f'Something wrong with {self._in_data_folder.get_file_path}')
        #     return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-folder', type=str,
                        default='/nfs/masi/xuk9/Data/VerSe2020/training_nii')
    parser.add_argument('--in-mask-folder', type=str,
                        default='/nfs/masi/xuk9/Data/VerSe2020/training_seg_nii')
    parser.add_argument('--out-clip-png-folder', type=str,
                        default='/nfs/masi/xuk9/Data/VerSe2020/training_clip')
    parser.add_argument('--file-list-txt', type=str,
                        default='/nfs/masi/xuk9/Data/VerSe2020/train_list')
    parser.add_argument('--out-clip-pptx', type=str,
                        default='/nfs/masi/xuk9/Data/VerSe2020/training_clip.pptx')
    parser.add_argument('--num-process', type=int, default=3)
    args = parser.parse_args()

    mkdir_p(args.in_mask_folder)
    mkdir_p(args.out_clip_png_folder)

    in_folder_obj = DataFolder(args.in_folder, args.file_list_txt)
    out_mask_folder_obj = DataFolder(args.in_mask_folder, args.file_list_txt)
    out_mask_folder_obj.change_suffix('_seg.nii.gz')
    out_clip_png_folder_obj = DataFolder(args.out_clip_png_folder, args.file_list_txt)
    out_clip_png_folder_obj.change_suffix('.png')

    exe_obj = ParalPPVMask(
        in_folder_obj,
        out_mask_folder_obj,
        out_clip_png_folder_obj,
        args.num_process
    )

    png_list = exe_obj.run_parallel()
    # save_pptx(png_list, args.out_clip_pptx)


# def main():
#     file_name = 'verse005.nii.gz'
#     # file_name = '00000001time20131205.nii.gz'
#
#     in_nii_folder = '/nfs/masi/xuk9/Data/VerSe2020/training_nii'
#     out_mask_nii_folder = '/nfs/masi/xuk9/Data/VerSe2020/training_seg_nii'
#     out_png_folder = '/nfs/masi/xuk9/Data/VerSe2020/training_clip'
#
#     mkdir_p(out_mask_nii_folder)
#     mkdir_p(out_png_folder)
#
#     in_nii = os.path.join(in_nii_folder, file_name)
#     out_nii = os.path.join(out_mask_nii_folder, file_name).replace('.nii.gz', '_seg.nii.gz')
#     out_png = os.path.join(out_png_folder, file_name.replace('.nii.gz', '.png'))
#
#     # get_pad_mask2(in_nii, out_nii)
#     multiple_clip_overlay_with_mask(in_nii, out_nii, out_png)
#     # clip_overlay_with_mask(in_nii, out_nii, out_png)

#
# if __name__ == '__main__':
#     main()
