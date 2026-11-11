#!/bin/bash
#SBATCH -J magex
#SBATCH -N 1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1
#SBATCH -t 04:00:00
# One MPI rank per core; keep BLAS single-threaded so ranks do not oversubscribe.
# For very large Wannier bases use fewer ranks with more threads (e.g. 10 x 4).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
srun magex run bccFe.toml          # or: mpirun -n 40 magex run bccFe.toml
