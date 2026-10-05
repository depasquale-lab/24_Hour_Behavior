#!/bin/bash -l

#$ -P depaqlab          # SCC project
#$ -l h_rt=24:00:00     # per-task wall clock; one rat per task
#$ -N tod_ddm           # job name
#$ -j y                 # merge stderr + stdout
#$ -pe omp 8            # threaded over the 5 CV folds + the all-data fit
#$ -t 1-18              # 18 24hr rats

cd "/projectnb/depaqlab/rsenne/hmmddm/24_Hour_Behavior/"

# Time-of-day DDM baseline, H = 0..3 harmonics, same folds as cv_k_24hr.
# Merge afterwards with: julia notebooks/FitTimeOfDayDDM.jl merge
# VARY_TAU=1 also lets τ follow time of day (outputs to results/tod_ddm_vary_tau/).
julia -t auto "notebooks/FitTimeOfDayDDM.jl"
