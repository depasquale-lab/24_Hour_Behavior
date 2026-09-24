#!/bin/bash -l

#$ -P depaqlab          # SCC project
#$ -l h_rt=96:00:00     # per-task wall clock; one rat per task
#$ -N ddmgrad_array     # job name
#$ -j y                 # merge stderr + stdout
#$ -pe omp 28           # 28 CPU cores per task
#$ -t 1-18              # one array task per 24hr rat (rat_names is 18 long)

cd "/projectnb/depaqlab/rsenne/hmmddm/24_Hour_Behavior/"

# -t auto lets Julia spin up one thread per core in the -pe allocation.
# SGE_TASK_ID is read by FitOneRatGradient.jl to pick which rat to fit.
#
# To fit a non-contiguous subset of rats, override -t and pass RAT_LIST.
# Use ':' as the separator — `qsub -v` treats ',' as a delimiter between vars.
#   qsub -t 1-5 -v RAT_LIST=5:12:14:15:18 run_directGradient_array.sh
# Or pre-export and use -V (commas are fine here):
#   RAT_LIST=5,12,14,15,18 qsub -V -t 1-5 run_directGradient_array.sh
# RAT_LIST values are 1-based rat indices; SGE_TASK_ID indexes into the list.
julia -t auto "notebooks/FitOneRatGradient.jl"
