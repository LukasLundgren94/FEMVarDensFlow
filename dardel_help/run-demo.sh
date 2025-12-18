#!/bin/sh
#SBATCH -J demo
#SBATCH -p main
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
# Number of MPI processes per node
#SBATCH -A naiss2025-5-251
#SBATCH --output slurm.out

ml PDCOLD/23.12  ;ml cpeGNU/23.12
ml spack-user/0.21.2

# Update with your own directory
export SPACK_USER_PREFIX=/cfs/klemming/projects/snic/naiss2024-6-123/lukas/spack-export-dir-pdcold/

 . /cfs/klemming/pdc/software/dardel/23.12/other/spack/0.21.2/user/share/spack/setup-env.sh

spack load fenics

# Update with your own path
python3 /cfs/klemming/projects/snic/naiss2024-6-123/lukas/fem_varden_flow_old/tests/repair_xdmf/repair_checkpoint.py results/checkpoint/*.xdmf

srun demo