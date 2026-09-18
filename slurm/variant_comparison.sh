#!/bin/bash
#SBATCH --job-name=variant_comparison
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

# Compare the mitoMAMMALmod GPR variants through module_1, once per OR strategy.
#
# calculate_ecs densifies the expression matrix and builds a per-gene dictionary from
# it, roughly two dense float32 copies, once per variant -- which is why this wants
# real memory and does not belong on a login node.
#
#   sbatch slurm/variant_comparison.sh
#   squeue -u $USER                     # watch it
#   tail -f variant_comparison-<jobid>.out
#
# --resume means a rerun scores whatever finished rather than starting over.

set -euo pipefail

BASE=/scratch/prj/crb_inner_ear/k2147692/metabolic
REPO=$BASE/code/Metabolic-pipeline
DATA=$BASE/data/kolla/kolla_E16.h5ad

module load python/3.11.6-gcc-13.2.0
source "$BASE/envs/metabolic/bin/activate"     # adjust if the venv lives elsewhere

cd "$REPO"
export MPLBACKEND=Agg

for STRATEGY in sum max; do
    echo "=============== or_strategy=$STRATEGY ==============="
    python -m metabolic_tools.model_variant_benchmark \
        --adata "$DATA" \
        --out "$BASE/results/05_variant_comparison_$STRATEGY" \
        --celltype-col cell_type \
        --and-strategy median \
        --or-strategy "$STRATEGY" \
        --resume
done

echo "done; summaries at $BASE/results/05_variant_comparison_{sum,max}/variant_comparison.tsv"
