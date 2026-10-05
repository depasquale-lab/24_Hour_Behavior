#!/bin/bash -l

#$ -P depaqlab          # SCC project
#$ -l h_rt=48:00:00     # per-task wall clock; one rat per task
#$ -N eddm_exact_daily  # job name
#$ -j y                 # merge stderr + stdout
#$ -pe omp 28           # marginal likelihood is threaded over trials
#$ -t 1-13              # 13 session-based rats (608840 dropped)

cd "/projectnb/depaqlab/rsenne/hmmddm/24_Hour_Behavior/"

# Full-data exact MLDDM fits for the session-based rats -> results/eddm_exact_daily/.
# Merge afterwards with: GROUP=daily julia notebooks/eDDMExact.jl merge
GROUP=daily julia -t $((NSLOTS-1)) "notebooks/eDDMExact.jl"
