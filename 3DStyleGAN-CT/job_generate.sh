#!/bin/bash
#rj name=3D-stylegan2-CT-Generate-noncon queue=hpi_ccta features=v100x4

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

python run_generator.py generate-images3D --network=results/00005-3dstylegan2-SCT_tfrecord256_2-4gpu-Gorig-Dres-R1-3d-MG-base882/network-snapshot-000732.pkl --seeds=100-110 --truncation-psi=0.3



echo "Test Done"
