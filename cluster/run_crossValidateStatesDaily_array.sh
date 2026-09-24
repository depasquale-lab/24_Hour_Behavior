#!/bin/bash -l

#$ -P depaqlab          # SCC project
#$ -l h_rt=96:00:00     # per-task wall clock; one (rat, fold) pair per task
#$ -N cv_k_daily        # job name
#$ -j y                 # merge stderr + stdout
#$ -pe omp 28           # 28 cores per task (threaded over random inits)
#$ -t 1-25              # 5 session-based rats x 5 folds, rat-major

cd "/projectnb/depaqlab/rsenne/hmmddm/24_Hour_Behavior/"

# SGE_TASK_ID indexes the (rat x fold) grid inside CrossValidateStatesDaily.jl.
# Each task sweeps K = 1..5 for its one held-out fold.
#
# To run every session-based animal instead of just the 5 already fit at K=4,
# set RAT_SET=all and change the array range to 1-70 (14 rats x 5 folds).
julia -t auto "notebooks/CrossValidateStatesDaily.jl"
