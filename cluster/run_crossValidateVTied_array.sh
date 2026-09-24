#!/bin/bash -l

#$ -P depaqlab          # SCC project
#$ -l h_rt=96:00:00     # per-task wall clock; one rat per task
#$ -N cv_vtied          # job name
#$ -j y                 # merge stderr + stdout
#$ -pe omp 28           # 28 cores per task (threaded over inits)
#$ -t 1-18              # one array task per 24hr rat

cd "/projectnb/depaqlab/rsenne/hmmddm/24_Hour_Behavior/"

julia -t auto "notebooks/CrossValidateVTied.jl"
