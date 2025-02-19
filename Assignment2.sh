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
#SBATCH --job-name=assignment2
#
# Output file
#SBATCH --output=assignment2.out
#======================================================

module purge

#Example module load command. 
#Load any modules appropriate for your program's requirements



#======================================================
# Prologue script to record job details
# Do not change the line below
#======================================================
/opt/software/scripts/job_prologue.sh  
#------------------------------------------------------

# Using the pylint tool to check formatting in the program 
pylint ./assignment2.py

# Run the program
./assignment2.py

#======================================================
# Epilogue script to record job endtime and runtime
# Do not change the line below
#======================================================
/opt/software/scripts/job_epilogue.sh 
#------------------------------------------------------
