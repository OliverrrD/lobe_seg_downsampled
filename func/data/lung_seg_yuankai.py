import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage
import nibabel as nib
import matplotlib.pyplot as plt
from glob import glob
import argparse


from skimage import measure, morphology
from PIL import Image
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage.morphology import binary_dilation,generate_binary_structure
from skimage.morphology import convex_hull_image
from scipy.ndimage.interpolation import zoom
from scipy.io import loadmat
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial


def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    if slices[0].ImagePositionPatient[2] == slices[1].ImagePositionPatient[2]:
        sec_num = 2;
        while slices[0].ImagePositionPatient[2] == slices[sec_num].ImagePositionPatient[2]:
            sec_num = sec_num + 1;
        slice_num = int(len(slices) / sec_num)
        slices.sort(key=lambda x: float(x.InstanceNumber))
        slices = slices[0:slice_num]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16), np.array([slices[0].SliceThickness] + slices[0].PixelSpacing,
                                                     dtype=np.float32)


def plot_3d(image, threshold=-300):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    verts, faces, _, _ = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


def binarize_per_slice(image, spacing, intensity_th=-600, sigma=1, area_th=30, eccen_th=0.99, bg_patch_size=10):
    bw = np.zeros(image.shape, dtype=bool)

    # prepare a mask, with all corner values set to nan
    image_size = image.shape[1]
    grid_axis = np.linspace(-image_size / 2 + 0.5, image_size / 2 - 0.5, image_size)
    x, y = np.meshgrid(grid_axis, grid_axis)
    d = (x ** 2 + y ** 2) ** 0.5
    nan_mask = (d < image_size / 2).astype(float)
    nan_mask[nan_mask == 0] = np.nan
    for i in range(image.shape[0]):
        # Check if corner pixels are identical, if so the slice  before Gaussian filtering
        if len(np.unique(image[i, 0:bg_patch_size, 0:bg_patch_size])) == 1:
            current_bw = scipy.ndimage.filters.gaussian_filter(np.multiply(image[i].astype('float32'), nan_mask), sigma,
                                                               truncate=2.0) < intensity_th
        else:
            current_bw = scipy.ndimage.filters.gaussian_filter(image[i].astype('float32'), sigma,
                                                               truncate=2.0) < intensity_th

        # select proper components
        label = measure.label(current_bw)
        properties = measure.regionprops(label)
        valid_label = set()
        for prop in properties:
            if prop.area * spacing[1] * spacing[2] > area_th and prop.eccentricity < eccen_th:
                valid_label.add(prop.label)
        current_bw = np.in1d(label, list(valid_label)).reshape(label.shape)
        bw[i] = current_bw

    return bw


def all_slice_analysis(bw, spacing, cut_num=0, vol_limit=[0.68, 8.2], area_th=6e3, dist_th=62):
    # in some cases, several top layers need to be removed first
    if cut_num > 0:
        bw0 = np.copy(bw)
        bw[-cut_num:] = False
    label = measure.label(bw, connectivity=1)
    # remove components access to corners
    mid = int(label.shape[2] / 2)
    bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], \
                    label[-1 - cut_num, 0, 0], label[-1 - cut_num, 0, -1], label[-1 - cut_num, -1, 0],
                    label[-1 - cut_num, -1, -1], \
                    label[0, 0, mid], label[0, -1, mid], label[-1 - cut_num, 0, mid], label[-1 - cut_num, -1, mid]])
    for l in bg_label:
        label[label == l] = 0

    # select components based on volume
    properties = measure.regionprops(label)
    for prop in properties:
        if prop.area * spacing.prod() < vol_limit[0] * 1e6 or prop.area * spacing.prod() > vol_limit[1] * 1e6:
            label[label == prop.label] = 0

    if len(np.unique(label)) == 1:
        return bw, 0

    # prepare a distance map for further analysis
    x_axis = np.linspace(-label.shape[1] / 2 + 0.5, label.shape[1] / 2 - 0.5, label.shape[1]) * spacing[1]
    y_axis = np.linspace(-label.shape[2] / 2 + 0.5, label.shape[2] / 2 - 0.5, label.shape[2]) * spacing[2]
    x, y = np.meshgrid(x_axis, y_axis)
    d = (x ** 2 + y ** 2) ** 0.5
    vols = measure.regionprops(label)
    valid_label = set()
    # select components based on their area and distance to center axis on all slices
    for vol in vols:
        single_vol = label == vol.label
        slice_area = np.zeros(label.shape[0])
        min_distance = np.zeros(label.shape[0])
        for i in range(label.shape[0]):
            slice_area[i] = np.sum(single_vol[i]) * np.prod(spacing[1:3])
            min_distance[i] = np.min(single_vol[i] * d + (1 - single_vol[i]) * np.max(d))

        if np.average([min_distance[i] for i in range(label.shape[0]) if slice_area[i] > area_th]) < dist_th:
            valid_label.add(vol.label)

    if len(valid_label) == 0:
        return bw, 0

    bw = np.in1d(label, list(valid_label)).reshape(label.shape)

    # fill back the parts removed earlier
    if cut_num > 0:
        # bw1 is bw with removed slices, bw2 is a dilated version of bw, part of their intersection is returned as final mask
        bw1 = np.copy(bw)
        bw1[-cut_num:] = bw0[-cut_num:]
        bw2 = np.copy(bw)
        bw2 = scipy.ndimage.binary_dilation(bw2, iterations=cut_num)
        bw3 = bw1 & bw2
        label = measure.label(bw, connectivity=1)
        label3 = measure.label(bw3, connectivity=1)
        l_list = list(set(np.unique(label)) - {0})
        valid_l3 = set()
        for l in l_list:
            indices = np.nonzero(label == l)
            l3 = label3[indices[0][0], indices[1][0], indices[2][0]]
            if l3 > 0:
                valid_l3.add(l3)
        bw = np.in1d(label3, list(valid_l3)).reshape(label3.shape)

    return bw, len(valid_label)


def fill_hole(bw):
    label = measure.label(~bw)
    bg_labels = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], \
                     label[-1, 0, 0], label[-1, 0, -1], label[-1, -1, 0], label[-1, -1, -1]])
    bw = ~np.in1d(label, list(bg_labels)).reshape(label.shape)

    return bw



def step1_python(case_path):
    case = load_scan(case_path)
    case_pixels, spacing = get_pixels_hu(case)
    bw = binarize_per_slice(case_pixels, spacing)
    flag = 0
    cut_num = 0
    cut_step = 2
    bw0 = np.copy(bw)
    while flag == 0 and cut_num < bw.shape[0]:
        bw = np.copy(bw0)
        bw, flag = all_slice_analysis(bw, spacing, cut_num=cut_num, vol_limit=[0.68,7.5])
        cut_num = cut_num + cut_step

    bw = fill_hole(bw)
    bw1, bw2, bw = two_lung_only(bw, spacing)
    return case_pixels, bw1, bw2, spacing


def two_lung_only(bw, spacing, max_iter=22, max_ratio=4.8):
    def extract_main(bw, cover=0.95):
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            properties.sort(key=lambda x: x.area, reverse=True)
            area = [prop.area for prop in properties]
            count = 0
            sum = 0
            while sum < np.sum(area) * cover:
                sum = sum + area[count]
                count = count + 1
            filter = np.zeros(current_slice.shape, dtype=bool)
            for j in range(count):
                bb = properties[j].bbox
                filter[bb[0]:bb[2], bb[1]:bb[3]] = filter[bb[0]:bb[2], bb[1]:bb[3]] | properties[j].convex_image
            bw[i] = bw[i] & filter

        label = measure.label(bw)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        bw = label == properties[0].label

        return bw

    def fill_2d_hole(bw):
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            for prop in properties:
                bb = prop.bbox
                current_slice[bb[0]:bb[2], bb[1]:bb[3]] = current_slice[bb[0]:bb[2], bb[1]:bb[3]] | prop.filled_image
            bw[i] = current_slice

        return bw

    found_flag = False
    iter_count = 0
    bw0 = np.copy(bw)
    while not found_flag and iter_count < max_iter:
        label = measure.label(bw, connectivity=2)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        if len(properties) > 1 and properties[0].area / properties[1].area < max_ratio:
            found_flag = True
            bw1 = label == properties[0].label
            bw2 = label == properties[1].label
        else:
            bw = scipy.ndimage.binary_erosion(bw)
            iter_count = iter_count + 1

    if found_flag:
        d1 = scipy.ndimage.morphology.distance_transform_edt(bw1 == False, sampling=spacing)
        d2 = scipy.ndimage.morphology.distance_transform_edt(bw2 == False, sampling=spacing)
        bw1 = bw0 & (d1 < d2)
        bw2 = bw0 & (d1 > d2)

        bw1 = extract_main(bw1)
        bw2 = extract_main(bw2)

    else:
        bw1 = bw0
        bw2 = np.zeros(bw.shape).astype('bool')

    bw1 = fill_2d_hole(bw1)
    bw2 = fill_2d_hole(bw2)
    bw = bw1 | bw2

    return bw1, bw2, bw

def step1_nifti(img_3d):
    img = img_3d.get_data()
    img2 = np.transpose(img, (2, 1, 0))
    case = img2[:, ::-1, :]
    case_pixels = np.array(case, dtype=np.int16)



    header=img_3d.header
    spacing=np.array([header['pixdim'][3]]+[header['pixdim'][2]]+[header['pixdim'][1]], dtype=np.float32)


    # case_pixels, spacing = get_pixels_hu(case)
    bw = binarize_per_slice(case_pixels, spacing)
    flag = 0
    cut_num = 0
    cut_step = 2
    bw0 = np.copy(bw)
    while flag == 0 and cut_num < bw.shape[0]:
        bw = np.copy(bw0)
        bw, flag = all_slice_analysis(bw, spacing, cut_num=cut_num, vol_limit=[0.68,7.5])
        cut_num = cut_num + cut_step

    bw = fill_hole(bw)
    bw1, bw2, bw = two_lung_only(bw, spacing)
    return case_pixels, bw1, bw2, spacing


def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1  = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1)>0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2)>2*np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3,1)
    dilatedMask = binary_dilation(convex_mask,structure=struct,iterations=10)
    return dilatedMask


def segment_a_lung(input_file,output_file):
    img_3d = nib.load(input_file)
    case_pixels, bw1, bw2, spacing = step1_nifti(img_3d)
    im, m1, m2, spacing = case_pixels, bw1, bw2, spacing
    # bw = binarize_per_slice(case_pixels, spacing)
    # bw3 = np.transpose(bw,(2, 1, 0))
    # bw3 = bw3[:,::-1,:]
    bw1 = np.transpose(bw1,(2, 1, 0))
    bw1 = bw1[:,::-1,:]
    bw2 = np.transpose(bw2,(2, 1, 0))
    bw2 = bw2[:,::-1,:]
    dm1 = process_mask(m1)
    dm2 = process_mask(m2)

    dilatedMask = dm1 + dm2
    Mask = m1 + m2
    extramask = dilatedMask ^ Mask

    ex3 = np.transpose(extramask,(2, 1, 0))
    ex3 = ex3[:,::-1,:]
    out = np.zeros(ex3.shape)
    out[ex3] = 6
    out[bw1] = 2
    out[bw2] = 4

    seg_img = nib.Nifti1Image(out.astype(int),img_3d.affine, img_3d.header)
    nib.save(seg_img, output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root_dir',default='/fs4/masi/huoy1/JeffFHSCT/raw_nifti_final_afterQA_correct/HeartScans/',help='input_root_dir')
    parser.add_argument('--filter',default='*.nii.gz',help='filter')
    parser.add_argument('--output_root_dir',default='/fs4/masi/huoy1/JeffFHSCT/raw_nifti_final_afterQA_correct/lung_segmentation/',help='output_root_dir')

    opt = parser.parse_args()
    print(opt)

    input_root_dir = opt.input_root_dir
    filter = opt.filter
    output_root_dir = opt.output_root_dir

    image_files = glob(os.path.join(input_root_dir, filter))
    image_files.sort()
    count = 0
    for input_img in image_files:
        count = count+1
        img_bname = os.path.basename(input_img)
        lung_seg_file = os.path.join(output_root_dir,img_bname)
        if not os.path.exists(lung_seg_file):
            segment_a_lung(input_img, lung_seg_file)
        print('%s done [%d/%d]\n'%(filter,count,len(image_files)))
#
# # patient_0_path = '/fs4/masi/huoy1/JeffFHSCT/tmp/testlungseg/1.3.12.2.1107.5.1.3.24033.4.0.1291106119762848'
# # case_pixels, bw1, bw2, spacing = step1_python(patient_0_path)
# # bw = binarize_per_slice(case_pixels, spacing)
#
#
# img_file = '/fs4/masi/huoy1/JeffFHSCT/tmp/testlungseg/s2PLT001-0002-00017-000064.nii.gz'
# seg_out_file = '/fs4/masi/huoy1/JeffFHSCT/tmp/testlungseg/mask_from_nifti6.nii.gz'
# segment_a_lung(img_file,seg_out_file)
#
#
# img_3d = nib.load(img_file)
# # img = img_3d.get_data()
# # plt.imshow(bw[0,:,:].astype(float), cmap=plt.cm.gray)
#
#
# case_pixels, bw1, bw2, spacing = step1_nifti(img_3d)
# im, m1, m2, spacing = case_pixels, bw1, bw2, spacing
# bw = binarize_per_slice(case_pixels, spacing)
# bw3 = np.transpose(bw,(2, 1, 0))
# bw3 = bw3[:,::-1,:]
#
# # case_pixels = np.transpose(case_pixels,(2, 1, 0))
# # case_pixels = case_pixels[:,::-1,:]
# bw1 = np.transpose(bw1,(2, 1, 0))
# bw1 = bw1[:,::-1,:]
# bw2 = np.transpose(bw2,(2, 1, 0))
# bw2 = bw2[:,::-1,:]
# #
# # plt.imshow(case_pixels[:,:,0].astype(float), cmap=plt.cm.gray)
# # plt.imshow(bw1[:,:,0].astype(float), cmap=plt.cm.gray)
# # plt.imshow(bw2[:,:,0].astype(float), cmap=plt.cm.gray)
#
# output_mask = '/fs4/masi/huoy1/JeffFHSCT/tmp/testlungseg/mask_from_nifti3.nii.gz'
# seg_img = nib.Nifti1Image(bw3.astype(float), img_3d.affine, img_3d.header)
# nib.save(seg_img, output_mask)
#
# output_mask = '/fs4/masi/huoy1/JeffFHSCT/tmp/testlungseg/mask_from_nifti5.nii.gz'
# seg_img = nib.Nifti1Image(bw1.astype(float), img_3d.affine, img_3d.header)
# nib.save(seg_img, output_mask)
#
# output_mask = '/fs4/masi/huoy1/JeffFHSCT/tmp/testlungseg/mask_from_nifti6.nii.gz'
# seg_img = nib.Nifti1Image(bw2.astype(float), img_3d.affine, img_3d.header)
# nib.save(seg_img, output_mask)
#
#
#
#
#
#
# dm1 = process_mask(m1)
# dm2 = process_mask(m2)
#
# dilatedMask = dm1 + dm2
# Mask = m1 + m2
# extramask = dilatedMask ^ Mask
#
# ex3 = np.transpose(extramask,(2, 1, 0))
# ex3 = ex3[:,::-1,:]
#
# output_mask = '/fs4/masi/huoy1/JeffFHSCT/tmp/testlungseg/mask_from_nifti4.nii.gz'
# seg_img = nib.Nifti1Image(ex3.astype(float), img_3d.affine, img_3d.header)
# nib.save(seg_img, output_mask)