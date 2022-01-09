#!/bin/bash
#rj name=3D-stylegan2-ENVcon256-TFD queue=hpi_ccta features=v100x4

module add anaconda3
#module add gcc/4.8.5
module add gcc/8.1.0
#module unload gcc-rt/9.2.0
#export PATH="/d/sw/anaconda3/5.2/bin:/d/sw/cuda/11.2/cuda-toolkit/bin:$PATH"
#echo $PATH
#export LD_LIBRARY_PATH="/usr/lib/x86_64-redhat-linux6E:$LD_LIBRARY_PATH"
#echo $LD_LIBRARY_PATH
#conda-env list
source activate 3dgan_tf2
#conda activate 3dgan_tf2

nvcc -V
nvidia-smi
python test.py
#gcc -v
#icpc -v
#which icpc

#nvcc test_nvcc.cu -o test_nvcc -run
#python dataset_tool_CT.py create_from_images3dCT256 datasets/SCT_tfrecord256_2 datasets/SCT12bit --shuffle 1 --base_size 8 8 2 
#python dataset_tool_CT.py create_from_images3dCT256 datasets/ENVcon_tfrecord256 datasets/ENVcon12bit --shuffle 1 --base_size 8 8 7 --vmin=-350 --vmax=1000

python run_training.py --num-gpus=4 --data-dir=datasets --config=Gorig-Dres-R1-3d-MG-base887-TFD --dataset=ENVcon12bit.txt --total-kimg=6000



echo "Test Done"
