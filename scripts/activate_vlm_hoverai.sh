#!/usr/bin/env bash

set -e

source /home/imit-learn/anaconda3/etc/profile.d/conda.sh
conda activate vlm_hoverai

# Keep the environment isolated from user-site packages during demo runs.
export PYTHONNOUSERSITE=1

echo "Activated conda environment: vlm_hoverai"
echo "PYTHONNOUSERSITE=1"
