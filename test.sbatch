#!/bin/bash -l
#SBATCH -p msibigmem  ### msibigmem, msilarge, msilong, amdlarge, amdsmall, msismall, amd512,amd2tb, agsmall SBATCH -p
#SBATCH --job-name=isd
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64          # match your SIMULATOR_CPU_COUNT
#SBATCH --mem=32G                      # or set e.g. --mem=64G or --mem-per-cpu=1G
#SBATCH --output=LOGS/output_%j.log
#SBATCH --error=LOGS/error_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=efe00002@umn.edu

set -euo pipefail
mkdir -p LOGS

# prevent over-threading inside each worker process
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# If you must pin to 3.6, call it directly; otherwise use your module/conda env
srun -n 1 /usr/bin/python3.11 -u /users/0/efe00002/multi_robot_multi_target_algorithms/batch_run.py
