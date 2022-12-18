#!/bin/bash

# Usage:
# lungReg.sh <ref image> <move image> <final dir> <final name (no .nii.gz)> <temp dir>

usage(){
echo "Usage: "
echo "  lungReg.sh <reference image> <moving image> <final dir> <final name (no .nii.gz)> <temp dir>"
echo "e.g. lungReg.sh ref.nii.gz move.nii.gz out.nii.gz ./"
echo 
echo "Outputs:"
echo " <output_directory>/<output_file>_regAladin.nii.gz - moving image rigid affine registered to the reference using NiftyReg"
echo " <output_directory>/<output_file>_regAladin.mat - affine transformation matrix from NiftyReg registration"
echo " <output_directory>/<output_file>_DeedsBCV_displacements.dat - deformation from Deeds registration"
echo " <output_directory>/ref_<reference image> - a copy from source directory, rename reference image with ref_<reference image>"
echo " <output_directory>/move_<move image> - a copy from source directory, rename moving image with move_<move image>"
echo " <output_directory>/final_<output_file>_DeedsBCV_deformed.nii.gz - FINAL RESULT - moving image non-rigid to reference"
}

if [ $# -ne 5 ]
then 
 echo "Error! Found $# arguments but expected 4"
 echo
 usage
 echo
 echo "Exitting."
 exit
fi

ref_image=$(readlink -f $1)
move_image=$(readlink -f $2)
mkdir -p $3
final_dir=$(readlink -f $3)
final_name=$(readlink -f $4)
mkdir -p $5
tmp_dir=$(readlink -f $5)

echo "output directory is ${find_dir}"

echo 'Step 1: Segment the lung and save the mask file and segmented file in tmp dir'
echo "ref image is ${ref_image}"
echo "move image is ${move_image}"
# /home/local/VANDERBILT/gaor2/anaconda3/envs/python36/bin/python /home/local/VANDERBILT/gaor2/code/RNNLung/func/tools/seg_lung.py --ori ${ref_image} --out ${tmp_dir}/ref_mask.nii.gz
# /home/local/VANDERBILT/gaor2/anaconda3/envs/python36/bin/python /home/local/VANDERBILT/gaor2/code/RNNLung/func/tools/seg_lung.py --ori ${move_image} --out ${tmp_dir}/move_mask.nii.gz
# /home/local/VANDERBILT/gaor2/anaconda3/envs/python36/bin/python /home/local/VANDERBILT/gaor2/code/RNNLung/func/tools/get_new_lung.py --ori ${ref_image} --mask ${tmp_dir}/ref_mask.nii.gz --out ${tmp_dir}/ref_seg.nii.gz
# /home/local/VANDERBILT/gaor2/anaconda3/envs/python36/bin/python /home/local/VANDERBILT/gaor2/code/RNNLung/func/tools/get_new_lung.py --ori ${move_image} --mask ${tmp_dir}/move_mask.nii.gz --out ${tmp_dir}/move_seg.nii.gz



echo "Step 2: Resample the segmented ref image and move image"


/fs4/masi/huoy1/Software/freesurfer6/freesurfer/bin/mri_convert ${tmp_dir}/ref_seg.nii.gz ${tmp_dir}/ref_resample.nii.gz -vs 1.0 1.0 1.0

/fs4/masi/huoy1/Software/freesurfer6/freesurfer/bin/mri_convert ${tmp_dir}/move_seg.nii.gz ${tmp_dir}/move_resample.nii.gz -vs 1.0 1.0 1.0

echo "step 3: Padding the image"

/home/local/VANDERBILT/gaor2/anaconda3/envs/python36/bin/python /home/local/VANDERBILT/gaor2/code/RNNLung/func/tools/padding.py --ori ${tmp_dir}/ref_resample.nii.gz --out ${tmp_dir}/ref_padding.nii.gz

/home/local/VANDERBILT/gaor2/anaconda3/envs/python36/bin/python /home/local/VANDERBILT/gaor2/code/RNNLung/func/tools/padding.py --ori ${tmp_dir}/move_resample.nii.gz --out ${tmp_dir}/move_padding.nii.gz

echo 'Step 4: LinearBCV affine to reference'
/home/local/VANDERBILT/gaor2/code/RNNLung/func/tools/linearBCVslow -F ${tmp_dir}/ref_padding.nii.gz -M ${tmp_dir}/move_padding.nii.gz -O ${tmp_dir}/linearBCV

echo 'Step 5: DeedsBCV non-rigid to reference'
/home/local/VANDERBILT/gaor2/code/RNNLung/func/tools/deedsBCVslow -F ${tmp_dir}/ref_padding.nii.gz -M ${tmp_dir}/move_padding.nii.gz -A ${tmp_dir}/linearBCV_matrix.txt -O ${final_dir}/${final_name}


