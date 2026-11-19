#!/bin/bash -l

#$ -P depaqlab          # SCC project
#$ -l h_rt=96:00:00     # per-task wall clock; one rat per task
#$ -N ddmgrad_array     # job name
#$ -j y                 # merge stderr + stdout
#$ -pe omp 28           # 28 CPU cores per task
#$ -t 1-18              # one array task per 24hr rat (rat_names is 18 long)

cd "/projectnb/depaqlab/rsenne/hmmddm/24_Hour_Behavior/"

# SGE_TASK_ID picks the rat in FitOneRatGradient.jl
# subset: qsub -t 1-5 -v RAT_LIST=5:12:14:15:18 run_directGradient_array.sh
# (':' since qsub -v splits on ',')
julia -t auto "notebooks/FitOneRatGradient.jl"
