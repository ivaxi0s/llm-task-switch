#!/bin/bash

#SBATCH -J inctxt
#SBATCH -A BYRNE-SL3-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#SBATCH -p ampere

numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e 's/^\([0-9][0-9]*\).*$/\1/')

. /etc/profile.d/modules.sh   # Leave this line (enables the module command)
module purge                  # Removes all modules still loaded
module load rhel8/default-amp # REQUIRED - loads the basic environment

# application="/home/ag2118/rds/hpc-work/inctxt/inctxt/scripts/eval_gw/single_gw-dm.sh"
application="/home/ag2118/rds/hpc-work/inctxt/inctxt/scripts/eval_dm/single_dm.sh"

#! Run options for the application:
options=""

#! Work directory (i.e. where the job will run):
workdir="$SLURM_SUBMIT_DIR" # The value of SLURM_SUBMIT_DIR sets workdir to the directory
# in which sbatch is run.

#! Are you using OpenMP (NB this is unrelated to OpenMPI)? If so increase this
#! safe value to no more than 128:
export OMP_NUM_THREADS=1

#! Number of MPI tasks to be started by the application per node and in total (do not change):
np=$((${numnodes} * ${mpi_tasks_per_node}))

#! Choose this for a pure shared-memory OpenMP parallel program on a single node:
#! (OMP_NUM_THREADS threads will be created):
#! CMD="$application $options"
CMD="$application $@"

cd $workdir
echo -e "Changed directory to $(pwd).\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: date"
echo "Running on master node: hostname"
echo "Current directory: pwd"

# Machine number and list of machines allocated to the job:
# Save machine file to sbatch/machine.file.$JOBID
MACHINEFILE=sbatch/machine.file.$JOBID
if [ "$SLURM_JOB_NODELIST" ]; then
  #! Create a machine file:
  export NODEFILE=$(generate_pbs_nodefile)
  cat $NODEFILE | uniq >$MACHINEFILE
  echo -e "\nNodes allocated:\n================"
  echo cat $MACHINEFILE | sed -e 's/\..*$//g'
fi

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

# Example usage:
# sbatch evaluate_gw-dm.sh --num_examples 4
