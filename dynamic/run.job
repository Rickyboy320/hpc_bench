#!/bin/sh
#SBATCH --time=00:15:00
#SBATCH -N 3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

#module load openmpi/gcc/64/1.10.3
module load mpich/ge/gcc/64/3.2
module load cuda10.0/toolkit/10.0.130

which mpirun
which mpiexec
#mpirun -n 2 valgrind --track-origins=yes ./vector.out "$@"
mpirun -n 3 ./vector.out "$@"
