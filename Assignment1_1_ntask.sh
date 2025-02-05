#!/bin/bash

#======================================================
#
# Job script for running a serial job on a single core 
#
#======================================================

#======================================================
# Propogate environment variables to the compute node
#SBATCH --export=ALL
#
# Run in the standard partition (queue)
#SBATCH --partition=teaching
#
# Specify project account
#SBATCH --account=teaching
#
# No. of tasks required (ntasks=1 for a single-core job)
#SBATCH --ntasks=1 --exclusive
#
# Specify (hard) runtime (HH:MM:SS)
#SBATCH --time=00:02:00
#
# Job name
#SBATCH --job-name=assignment1
#
# Output file
#SBATCH --output=pi_approx_ntasks_1.out
#======================================================

module purge

#Example module load command. 
#Load any modules appropriate for your program's requirements

module load fftw/gcc-8.5.0/3.3.10
module load openmpi/gcc-8.5.0/4.1.1

#======================================================
# Prologue script to record job details
# Do not change the line below
#======================================================
/opt/software/scripts/job_prologue.sh  
#------------------------------------------------------

# Using the pylint tool to check formatting in the program 
pylint --extension-pkg-allow-list=mpi4py.MPI assignment1.py

# Run the program
mpirun -np $SLURM_NPROCS ./assignment1.py

#======================================================
# Epilogue script to record job endtime and runtime
# Do not change the line below
#======================================================
/opt/software/scripts/job_epilogue.sh 
#------------------------------------------------------
