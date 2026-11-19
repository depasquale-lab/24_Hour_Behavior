#!/bin/bash -l

#$ -P depaqlab          # SCC project
#$ -l h_rt=48:00:00     # per-task wall clock; one (rat, rep) per task
#$ -N mlddm_recovery    # job name
#$ -j y                 # merge stderr + stdout
#$ -pe omp 28           # marginal likelihood is threaded over trials
#$ -t 1-18              # 18 24hr rats x N_REP=1; raise with N_REP and -t 1-(18*N_REP)

cd "/projectnb/depaqlab/rsenne/hmmddm/24_Hour_Behavior/"

# SGE_TASK_ID indexes TASKS in notebooks/MLDDMRecovery.jl (rep-major).
# Plot afterwards with: julia notebooks/PlotMLDDMRecovery.jl
julia -t auto "notebooks/MLDDMRecovery.jl"
