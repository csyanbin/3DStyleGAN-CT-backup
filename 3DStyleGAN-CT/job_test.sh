#!/bin/bash
#rj name=3D-stylegan2-CT queue=hpi_ccta features=v100x4

module add anaconda3
#module add gcc/4.8.5
module add gcc/8.1.0
#module unload gcc-rt/9.2.0
#export PATH="/d/sw/anaconda3/5.2/bin:/d/sw/cuda/11.2/cuda-toolkit/bin:$PATH"
#echo $PATH
#export LD_LIBRARY_PATH="/usr/lib/x86_64-redhat-linux6E:$LD_LIBRARY_PATH"
#echo $LD_LIBRARY_PATH
#conda-env list
export LD_LIBRARY_PATH="/data/hpi_ccta/yanbin/.local/lib:$LD_LIBRARY_PATH"
source activate TF1
#conda activate 3dgan_tf2

nvcc -V
nvidia-smi
python test.py
gcc -v
icpc -v
which icpc

nvcc test_nvcc.cu -o test_nvcc -run




echo "Test Done"
