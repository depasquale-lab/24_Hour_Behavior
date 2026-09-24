#!/bin/bash -l

#$ -P depaqlab          # SCC project
#$ -l h_rt=48:00:00     # per-task wall clock
#$ -N ddmhmm_recovery   # job name
#$ -j y                 # merge stderr + stdout
#$ -pe omp 28           # 20 blind inits + 1 truth-initialised fit run in parallel

# One recovery task per job; TASK indexes TASKS in ParameterRecovery.jl.
# Submit all 90 with:
#   for t in $(seq 1 72); do qsub -o logs/ -v TASK=$t run_parameterRecovery.sh; done

module load julia/1.11.7
cd "/projectnb/depaqlab/rsenne/hmmddm/24_Hour_Behavior/"

julia -t auto "notebooks/ParameterRecovery.jl" "$TASK"
