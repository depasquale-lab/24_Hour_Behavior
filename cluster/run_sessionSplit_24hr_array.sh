#!/bin/bash -l

#$ -P depaqlab          # SCC project
#$ -l h_rt=48:00:00     # per-task wall clock; one rat per task
#$ -N split_24hr        # job name
#$ -j y                 # merge stderr + stdout
#$ -pe omp 16           # threaded over inits / sessions
#$ -t 1-18              # rats

cd "/projectnb/depaqlab/rsenne/hmmddm/24_Hour_Behavior/"

# Within-session split: DDM vs per-session DDM vs DDM-HMM on held-out session ends.
# Merge with: GROUP=24hr julia notebooks/SessionSplitDDM.jl merge
GROUP=24hr julia -t $((NSLOTS-1)) "notebooks/SessionSplitDDM.jl"
