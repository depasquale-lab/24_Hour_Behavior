#!/bin/bash -l
# Submit FitOneRatGradient.jl for an arbitrary subset of rats.
# Generates a tempfile with the #$ -t and #$ -v directives baked in,
# so this works even if the local qsub rejects -t / -v on the command line.
#
# Usage:
#   cluster/submit_directGradient_subset.sh 5 12 14 15 18
#   cluster/submit_directGradient_subset.sh 5,12,14,15,18
#   cluster/submit_directGradient_subset.sh 5:12:14:15:18

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <rat indices, space/comma/colon separated>"
    echo "Examples:"
    echo "  $0 5 12 14 15 18"
    echo "  $0 5,12,14,15,18"
    exit 1
fi

# Normalize args (any of space/comma/colon) into a colon-separated list.
RAT_LIST=$(echo "$@" | tr ' ,' ':' | tr -s ':')
N=$(echo "$RAT_LIST" | awk -F: '{print NF}')

if [[ "$N" -lt 1 ]]; then
    echo "ERROR: parsed 0 rat indices from '$*'"
    exit 1
fi

WORKDIR="/projectnb/depaqlab/rsenne/hmmddm/24_Hour_Behavior"

TMP=$(mktemp "${WORKDIR}/.ddmgrad_subset.XXXXXX.sh")
trap 'rm -f "$TMP"' EXIT

cat > "$TMP" <<EOF
#!/bin/bash -l

#\$ -P depaqlab
#\$ -l h_rt=96:00:00
#\$ -N ddmgrad_subset
#\$ -j y
#\$ -pe omp 28
#\$ -t 1-${N}
#\$ -v RAT_LIST=${RAT_LIST}

cd "${WORKDIR}/"
julia -t auto "notebooks/FitOneRatGradient.jl"
EOF

echo "Submitting subset job: ${N} task(s) over rats ${RAT_LIST//:/ }"
qsub "$TMP"
