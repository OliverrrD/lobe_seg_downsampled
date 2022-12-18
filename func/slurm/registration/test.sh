#!/bin/bash
#SBATCH --job-name=test_slurm
#SBATCH --output=./out/out.txt
#SBATCH --ntasks=1
#SBATCH --time=03:30:00
#SBATCH --mem-per-cpu=12G

sh /home/local/VANDERBILT/gaor2/code/RNNLung/func/tools/REGLUNG.sh /share3/gaor2/share5backup/tmp/reg_test4/ref_363532318time20150706item00.nii.gz /share3/gaor2/share5backup/tmp/reg_test4/move_363532318time20140602item00.nii.gz /share3/gaor2/share5backup/tmp/reg_test4/slurm_final final_out_sessname /share3/gaor2/share5backup/tmp/reg_test4/slurm_tmp

date
