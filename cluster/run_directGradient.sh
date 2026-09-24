#!/bin/bash -l

#$ -P depaqlab
#$ -l h_rt=12:00:00
#$ -N ddm_grad
#$ -j y
#$ -pe omp 28

cd "/projectnb/depaqlab/rsenne/hmmddm/24_Hour_Behavior/"

julia -t auto "notebooks/DirectGradientDDMHMM.jl"
