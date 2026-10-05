#!/bin/bash -l

#$ -P depaqlab          # SCC project
#$ -l h_rt=48:00:00     # per-task wall clock; one (rat, fold) pair per task
#$ -N cv_shuf_24hr      # job name
#$ -j y                 # merge stderr + stdout
#$ -pe omp 16           # threaded over random inits
#$ -t 1-90              # rats x 5 folds, rat-major

cd "/projectnb/depaqlab/rsenne/hmmddm/24_Hour_Behavior/"

# K=4 fit on within-session-shuffled training sessions (mixture-of-DDMs control),
# scored on the real held-out sessions -> results/ddm_hmm_state_sweep/cv*_shuffled/.
# Merge with: GROUP=24hr SHUFFLE=1 julia notebooks/MergeStateSweepCV.jl
GROUP=24hr RAT_SET=all SHUFFLE=1 K_LIST=4 julia -t $((NSLOTS-1)) "notebooks/CrossValidateStatesDaily.jl"
