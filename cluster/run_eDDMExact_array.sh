#!/bin/bash -l

#$ -P depaqlab          # SCC project
#$ -l h_rt=48:00:00     # per-task wall clock; one rat per task
#$ -N eddm_exact        # job name
#$ -j y                 # merge stderr + stdout
#$ -pe omp 28           # marginal likelihood is threaded over trials
#$ -t 1-18              # 18 24hr rats

cd "/projectnb/depaqlab/rsenne/hmmddm/24_Hour_Behavior/"

# SGE_TASK_ID indexes the rats in notebooks/eDDMExact.jl.
# Merge afterwards with: julia notebooks/eDDMExact.jl merge
julia -t auto "notebooks/eDDMExact.jl"
