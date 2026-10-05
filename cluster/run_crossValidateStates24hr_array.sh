#!/bin/bash -l

#$ -P depaqlab          # SCC project
#$ -l h_rt=96:00:00     # per-task wall clock; one (rat, fold) pair per task
#$ -N cv_k_24hr         # job name
#$ -j y                 # merge stderr + stdout
#$ -pe omp 28           # 28 cores per task (threaded over random inits)
#$ -t 1-90              # 18 24hr rats x 5 folds, rat-major

cd "/projectnb/depaqlab/rsenne/hmmddm/24_Hour_Behavior/"

# Same K = 1..5 CV sweep as run_crossValidateStatesDaily_array.sh, on the 24 hr
# animals (sessions = calendar days). Outputs go to results/ddm_hmm_state_sweep/cv_24hr/.
GROUP=24hr julia -t auto "notebooks/CrossValidateStatesDaily.jl"
