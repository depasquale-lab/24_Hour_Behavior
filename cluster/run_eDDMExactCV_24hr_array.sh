#!/bin/bash -l

#$ -P depaqlab          # SCC project
#$ -l h_rt=48:00:00     # per-task wall clock; one (rat, fold) pair per task
#$ -N eddm_cv_24hr      # job name
#$ -j y                 # merge stderr + stdout
#$ -pe omp 16           # marginal likelihood is threaded over trials
#$ -t 1-90              # 18 24hr rats x 5 folds, rat-major

cd "/projectnb/depaqlab/rsenne/hmmddm/24_Hour_Behavior/"

# Held-out MLDDM scoring on the DDM-HMM CV folds. Needs the full-data fits first.
# Merge afterwards with: GROUP=24hr julia notebooks/eDDMExact.jl cv-merge
GROUP=24hr julia -t $((NSLOTS-1)) "notebooks/eDDMExact.jl" cv
