#!/bin/bash -l

#$ -P depaqlab          # SCC project
#$ -l h_rt=96:00:00     # per-task wall clock; one rat per task
#$ -N ddmhmm_array      # job name
#$ -j y                 # merge stderr + stdout
#$ -pe omp 28           # 28 CPU cores per task
#$ -t 1-18              # one array task per 24hr rat (rat_names is 18 long)

cd "/projectnb/depaqlab/rsenne/hmmddm/24_Hour_Behavior/"

# -t auto lets Julia spin up one thread per core in the -pe allocation.
# SGE_TASK_ID is read by FitOneRat.jl to pick which rat to fit.
julia -t auto "notebooks/FitOneRat.jl"
