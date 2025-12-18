#!/bin/sh
#SBATCH -J install
#SBATCH -p shared
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=10
#SBATCH -A naiss2025-1-5
#SBATCH --output slurm.out

ml PDCOLD/23.12 ;  ml cpeGNU/23.12
ml spack-user/0.21.2

# Run this script in one of the test directories (for example varden/tests/ocean)

# replace with your own Spack directory
export SPACK_USER_PREFIX=/cfs/klemming/projects/snic/naiss2024-6-123/lukas/spack-export-dir-pdcold/

 . /cfs/klemming/pdc/software/dardel/23.12/other/spack/0.21.2/user/share/spack/setup-env.sh

spack load fenics

# replace with your own directory
source /cfs/klemming/projects/snic/naiss2024-6-123/lukas/fem_varden_flow_old/fem_varden_flow.conf

orig_dir=$(pwd)

cd ../../../build;
cmake ..;
make -j 10 install;

# Replace with your own test directory
cd "$orig_dir";
cmake .;
make -j 10;
