#!/bin/sh
#SBATCH -J idealized-20km
#SBATCH -p cpu
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH -A naiss2026-1-3-cpu
#SBATCH --output=slurm.out

set -eu

# print some info about the job
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Number of tasks per node: $SLURM_TASKS_PER_NODE"
echo "Total number of tasks: $SLURM_NTASKS"

# record job start time
start_time=$(date +%s)
start_time_human=$(date -u +"%Y-%m-%d %H:%M:%S %Z")
echo "Start time: $start_time_human"

SPACK_ROOT=${SPACK_ROOT:-/home/jonwisk/_WORK/spack}
SPACK_ENV_DIR=${SPACK_ENV_DIR:-$SPACK_ROOT/var/spack/environments/fem-varden-flow}
. "$SPACK_ROOT/share/spack/setup-env.sh"
spack env activate "$SPACK_ENV_DIR"

REPAIR_DIR=../../tests/repair_xdmf

repair_files() {
	repair_script=$1
	shift
	for xdmf_file do
		[ -f "$xdmf_file" ] || continue
		python3 "$REPAIR_DIR/$repair_script" "$xdmf_file"
	done
}

repair_files repair_checkpoint.py results/checkpoint/*.xdmf

srun ./demo

repair_files repair_xdmf.py results/*.xdmf
repair_files repair_checkpoint.py results/checkpoint/*.xdmf
for xdmf_file in results/fine/*.xdmf; do
	[ -f "$xdmf_file" ] || continue
	case "$xdmf_file" in
		*avg*.xdmf) repair_files repair_xdmf_avg.py "$xdmf_file" ;;
		*) repair_files repair_xdmf.py "$xdmf_file" ;;
	esac
done

# move mesh and parameter files to results directory
cp parameters.json results/fine/.
cp run.sh results/fine/.
cp slurm.out results/fine/.
cp *.h5 results/fine/.
cp *.xdmf results/fine/.
cp *.msh results/fine/.
cp time_history.m results/fine/.

# rename fine directory to avoid overwriting previous results using jobnumber
mv results/fine results/fine-$(printf '%06d' "$SLURM_JOB_ID")
mv results/checkpoint results/checkpoint-$(printf '%06d' "$SLURM_JOB_ID")
