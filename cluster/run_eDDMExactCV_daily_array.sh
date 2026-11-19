#!/bin/bash -l

#$ -P depaqlab          # SCC project
#$ -l h_rt=48:00:00     # per-task wall clock; one (rat, fold) pair per task
#$ -N eddm_cv_daily      # job name
#$ -j y                 # merge stderr + stdout
#$ -pe omp 16           # marginal likelihood is threaded over trials
#$ -t 1-65              # 13 session-based rats x 5 folds, rat-major

cd "/projectnb/depaqlab/rsenne/hmmddm/24_Hour_Behavior/"

# held-out MLDDM on the DDM-HMM CV folds; needs full-data fits first
# merge: GROUP=daily julia notebooks/eDDMExact.jl cv-merge
GROUP=daily julia -t $((NSLOTS-1)) "notebooks/eDDMExact.jl" cv
