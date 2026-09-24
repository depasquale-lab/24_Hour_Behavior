#!/bin/bash -l

#$ -P depaqlab       # Specify the SCC project name you want to use
#$ -l h_rt=96:00:00   # Specify the hard time limit for the job
#$ -N ddmhmms           # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -pe omp 28        # Request 28 CPU cores


module load julia/1.11.7
cd "/projectnb/depaqlab/rsenne/hmmddm/24_Hour_Behavior/"
julia "notebooks/FitDDMHMMs.jl"


