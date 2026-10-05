#!/bin/bash -l

#$ -P depaqlab          # SCC project
#$ -l h_rt=24:00:00     # merges + summary + ACF simulations + figure
#$ -N cohort_final      # job name
#$ -j y                 # merge stderr + stdout
#$ -pe omp 16           # ACF simulations are threaded over replicates

cd "/projectnb/depaqlab/rsenne/hmmddm/24_Hour_Behavior/"

# Submit held on every session-cohort job (see notebooks/README.md). Merges all
# per-task outputs, then builds results/session_cohort/ and the figure.
set -x
for g in daily 24hr; do
    GROUP=$g julia notebooks/MergeStateSweepCV.jl
    GROUP=$g SHUFFLE=1 julia notebooks/MergeStateSweepCV.jl
    GROUP=$g julia notebooks/eDDMExact.jl cv-merge
    GROUP=$g julia notebooks/SessionSplitDDM.jl merge
done
GROUP=daily julia notebooks/eDDMExact.jl merge
julia -t $((NSLOTS-1)) notebooks/SessionCohortSummary.jl
julia notebooks/PlotSessionCohortFigure.jl
