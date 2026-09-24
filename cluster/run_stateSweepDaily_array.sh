#!/bin/bash -l

#$ -P depaqlab          # SCC project
#$ -l h_rt=96:00:00     # per-task wall clock; one rat per task
#$ -N k_sweep_daily     # job name
#$ -j y                 # merge stderr + stdout
#$ -pe omp 28           # 28 cores per task (threaded over inits)
#$ -t 1-14              # one array task per session-based ("daily") rat

cd "/projectnb/depaqlab/rsenne/hmmddm/24_Hour_Behavior/"

# SGE_TASK_ID is read by StateSweepDaily.jl to pick which rat to fit.
# Override the K grid with e.g. K_LIST=1,2,3,4,5,6,7,8
julia -t auto "notebooks/StateSweepDaily.jl"
