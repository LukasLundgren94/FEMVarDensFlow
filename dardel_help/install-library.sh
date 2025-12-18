#!/bin/sh
#SBATCH -J install
#SBATCH -p main
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
# Number of MPI processes per node

#SBATCH -A naiss2025-1-5
#SBATCH --output slurm.out
ml PDCOLD/23.12 ;ml cpeGNU/23.12
ml spack-user/0.21.2

# Run this script from the main repo folder

# Update with your own spack path
export SPACK_USER_PREFIX=/cfs/klemming/projects/snic/naiss2024-6-123/lukas/spack-export-dir-pdcold/

 . /cfs/klemming/pdc/software/dardel/23.12/other/spack/0.21.2/user/share/spack/setup-env.sh

spack load fenics

# Replace with your own path
source /cfs/klemming/projects/snic/naiss2024-6-123/lukas/fem_varden_flow_old/fem_varden_flow.conf

cd core/src/ufl/norm/
python3 generate_ufl.py *.ufl.src
./compile_forms *.ufl

cd ../pdesmooth
python3 generate_ufl.py *.ufl.src
./compile_forms *.ufl

cd ../../../../varden/src/solver/ufl
python3 generate_ufl.py *.ufl.src
./compile_forms *.ufl

cd ../../common/ufl
python3 generate_ufl.py *.ufl.src
./compile_forms *.ufl

cd ../../../../

mkdir build
cd build
cmake ..
cmake --build . --target core -j 128
cmake --build . --target ocean -j 128
cmake --build . --target varden -j 128

make -j 128 install 
