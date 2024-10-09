#!/bin/sh
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -n 32
#SBATCH --ntasks-per-node 4
#SBATCH --gpus-per-task 1
#SBATCH -t 4:00:00
#SBATCH -A atlas
#SBATCH --gpu-bind=none
#SBATCH --image=vmikuni/tensorflow:ngc-23.12-tf2-v1
#SBATCH --module=gpu,nccl-2.18

export TF_CPP_MIN_LOG_LEVEL=2
echo srun --mpi=pmi2 shifter python train.py --dataset tau --folder tau --lr 3e-5 --layer_scale --local --mode classifier --warm_epoch 3 --epoch 30 --stop_epoch 3 --batch 256
srun --mpi=pmi2 shifter python train.py --dataset tau --folder tau --lr 3e-5 --layer_scale --local --mode classifier --warm_epoch 3 --epoch 30 --stop_epoch 3 --batch 256
