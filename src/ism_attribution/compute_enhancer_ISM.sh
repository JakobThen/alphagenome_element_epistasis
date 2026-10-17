#!/usr/bin/env bash
# Compute AlphaGenome ISM attributions for every enhancer in the BED file and
# write data/processed/K562_ISM__enhancers.h5.
#
# Run from the repository root with the project environment active:
#     export ALPHAGENOME_API_KEY=...        # or put it in ~/.alphagenome_api_key
#     bash src/ism_attribution/compute_enhancer_ISM.sh
#
# ISM scores every base of every region through the API, so a full run takes
# hours. On a cluster, submit this script through your scheduler unchanged.

set -euo pipefail

# Inputs and output, relative to the repository root
INPUT_BED="data/raw/enhancers.bed"
TFS="data/raw/tfs.tsv"
TARGET_GENES="data/raw/target_genes.csv"
OUT_PREFIX="data/processed/K562_ISM_"

CURIES="EFO:0002067"      # K562
CONTEXT_LENGTH="1MB"      # one of 16KB, 100KB, 500KB, 1MB

API_KEY="${ALPHAGENOME_API_KEY:-$(cat "${HOME}/.alphagenome_api_key")}"

mkdir -p "$(dirname "${OUT_PREFIX}")"

echo "Started: $(date)"

python src/ism_attribution/run_ism.py \
    --bed "${INPUT_BED}" \
    --out_prefix "${OUT_PREFIX}" \
    --curies "${CURIES}" \
    --tfs "${TFS}" \
    --genes "${TARGET_GENES}" \
    --seq_len "${CONTEXT_LENGTH}" \
    --api_key "${API_KEY}"

echo "Finished: $(date)"
