#!/bin/bash -l

#$ -P depaqlab          # SCC project
#$ -l h_rt=96:00:00     # per-task wall clock; one (rat, fold) pair per task
#$ -N cv_k_daily        # job name
#$ -j y                 # merge stderr + stdout
#$ -pe omp 28           # 28 cores per task (threaded over random inits)
#$ -t 1-65              # 13 session-based rats x 5 folds, rat-major (608840 dropped)

cd "/projectnb/depaqlab/rsenne/hmmddm/24_Hour_Behavior/"

# K = 1..5 sweep, one held-out fold per task
# RAT_SET=all: rats with >= 5 sessions; SKIP_DONE=1 skips finished tasks
RAT_SET=all SKIP_DONE=1 julia -t $((NSLOTS-1)) "notebooks/CrossValidateStatesDaily.jl"
