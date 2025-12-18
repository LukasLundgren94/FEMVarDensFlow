#!/bin/sh
#SBATCH -J install
#SBATCH -p main
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH -A naiss2025-5-251
#SBATCH --output slurm.out

ml PDCOLD/23.12
ml spack-user/0.21.2

# Replace with your own directory (you only need to create it, nothing more)
export SPACK_USER_PREFIX=/cfs/klemming/projects/snic/naiss2024-6-123/lukas/spack-export-dir-pdcold/

# For some reason PDCOLD/23.12 only allows compilation in serial with spack...
spack install  fenics%gcc@12.3.0 cflags="-O3" fflags="-O3" openmp==False ++zlib ^python@3.10.13 ^boost@1.83.0 ^openblas ^petsc+suite-sparse+mumps

# To check if everything installed properly
ffc --version


# PDC/24.11 and Spack doesn't work yet
# ml PDC/24.11
# ml cpeGNU/24.11
# ml spack-user/0.23.1
# ml gcc-native/12.3
# ml gcc-native/13.2
#ml cray-mpich/8.1.28