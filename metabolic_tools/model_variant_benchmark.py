"""
Run the module_1 path once per model variant and measure whether the ECS embedding
holds its shape.

The question this answers is the one that decides whether a GPR patch is safe to
publish: adding isozymes to OR rules adds ECS features (split_isozymes gives every
OR branch its own column), and features that carry only ambient signal are noise
dimensions that PCA and the neighbour graph still have to absorb. If cell types stop
separating, the patch has cost more than it bought -- a blobbier UMAP is worse.

Nothing here looks at which genes a particular tissue expresses, so the verdict
stays a property of the model rather than of the dataset it was measured on. Run it
on more than one dataset before drawing a conclusion.

Metrics, all computed on the embedding module_1 itself produced:

    n_features          ECS columns surviving deduplication
    silhouette_pca      cell-type silhouette on the 40 PCs the neighbour graph uses
    knn_purity          fraction of each cell's 15 nearest neighbours sharing its type
    ari / nmi           leiden against the cell-type labels
    umap_separation     between-centroid distance over within-type spread, on the UMAP
    mean_abs_residual   mean |Pearson residual| of the leiden x cell-type table

silhouette_pca, knn_purity and umap_separation all fall as the embedding blurs;
n_features rising while they fall is the specific failure mode to watch for.

features_added/lost_vs_baseline count label changes as well as genuinely new columns,
because ecs_calculator names a split isozyme feature after the common prefix of the
genes in its branch -- adding Ldhc to an Ldha/Ldhb branch renames it. Read the
n_features delta for how much the feature space actually grew.

Usage
-----
    python -m metabolic_tools.model_variant_benchmark \\
        --adata data/kolla_e16_metacells.h5ad --out results/variant_comparison
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from .cleaning_report import cleaning_report
from .ecs_calculator import calculate_ecs
from .module_1 import celltype_annotate

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEFAULT_VARIANTS = {
    "baseline": os.path.join(DATA_DIR, "mitoMAMMALmod.json"),
    "repaired": os.path.join(DATA_DIR, "mitoMAMMALmod_repaired.json"),
    "conservative": os.path.join(DATA_DIR, "mitoMAMMALmod_conservative.json"),
    "inclusive": os.path.join(DATA_DIR, "mitoMAMMALmod_inclusive.json"),
}


def embedding_metrics(adata, celltype_col="majority_celltype", n_pcs=40, n_neighbors=15):
    """Cell-type structure in an annotated ECS embedding, as a flat dict."""
    labels = adata.obs[celltype_col].astype(str).values
    codes = pd.Categorical(labels).codes
    metrics = {"n_cells": adata.n_obs, "n_features": adata.n_vars,
               "n_celltypes": len(set(labels))}

    pcs = adata.obsm["X_pca"][:, :n_pcs]
    metrics["silhouette_pca"] = float(silhouette_score(pcs, codes)) if len(set(codes)) > 1 else np.nan

    # kNN purity off the same connectivity graph module_1 clustered on
    if "distances" in adata.obsp:
        graph = adata.obsp["distances"].tolil()
        shares = []
        for i in range(adata.n_obs):
            neighbours = graph.rows[i][:n_neighbors]
            if neighbours:
                shares.append(np.mean([codes[j] == codes[i] for j in neighbours]))
        metrics["knn_purity"] = float(np.mean(shares)) if shares else np.nan
    else:
        metrics["knn_purity"] = np.nan

    if "leiden" in adata.obs:
        leiden = adata.obs["leiden"].astype(str).values
        metrics["ari"] = float(adjusted_rand_score(labels, leiden))
        metrics["nmi"] = float(normalized_mutual_info_score(labels, leiden))
        metrics["n_clusters"] = int(pd.Series(leiden).nunique())
        table = pd.crosstab(adata.obs["leiden"], adata.obs[celltype_col])
        expected = np.outer(table.sum(1), table.sum(0)) / table.values.sum()
        residuals = (table.values - expected) / np.sqrt(expected)
        metrics["mean_abs_residual"] = float(np.abs(residuals).mean())

    # UMAP separation: how far apart the cell-type centroids sit relative to the
    # spread within each type. Falls when types smear into one blob.
    if "X_umap" in adata.obsm:
        umap = adata.obsm["X_umap"]
        centroids, spreads = [], []
        for code in np.unique(codes):
            points = umap[codes == code]
            centre = points.mean(axis=0)
            centroids.append(centre)
            spreads.append(np.linalg.norm(points - centre, axis=1).mean())
        centroids = np.array(centroids)
        pairwise = [np.linalg.norm(a - b)
                    for i, a in enumerate(centroids) for b in centroids[i + 1:]]
        metrics["umap_separation"] = float(np.mean(pairwise) / np.mean(spreads)) if spreads else np.nan
    return metrics


def run_variant(name, model_path, adata_path, out_dir, celltype_col="majority_celltype",
                species="mmusculus", gene_column="gene_symbol",
                and_strategy="median", or_strategy="sum", split_isozymes=True,
                cluster_resolution=1.0):
    """cleaning_report -> ECS -> module_1 for one model, returning its metrics."""
    variant_dir = os.path.join(out_dir, name)
    os.makedirs(variant_dir, exist_ok=True)
    print(f"\n########## {name} ##########")

    cleaning_report(adata_path=adata_path, model_path=model_path, species=species,
                    gene_column=gene_column, output_dir=variant_dir)
    report_path = next(os.path.join(variant_dir, f) for f in sorted(os.listdir(variant_dir))
                       if f.endswith(".json"))

    calculate_ecs(adata_path=adata_path, model_path=model_path,
                  cleaning_report_path=report_path, output_dir=variant_dir,
                  symbol_col=gene_column, split_isozymes=split_isozymes,
                  and_strategy=and_strategy, or_strategy=or_strategy)
    ecs_path = os.path.join(variant_dir, "ecs_matrix_isozyme_split.h5ad")

    annotated = celltype_annotate(adata_path=ecs_path, output_dir=variant_dir,
                                  celltype_col=celltype_col,
                                  cluster_resolution=cluster_resolution)
    metrics = embedding_metrics(sc.read_h5ad(annotated), celltype_col)
    metrics["variant"] = name
    metrics["model"] = os.path.basename(model_path)
    return metrics, ecs_path


def compare_variants(adata_path, out_dir, variants=None, **kwargs):
    """Run every variant and write a side-by-side summary."""
    variants = variants or DEFAULT_VARIANTS
    os.makedirs(out_dir, exist_ok=True)
    rows, feature_sets = [], {}
    for name, model_path in variants.items():
        metrics, ecs_path = run_variant(name, model_path, adata_path, out_dir, **kwargs)
        rows.append(metrics)
        feature_sets[name] = set(sc.read_h5ad(ecs_path).var_names)

    summary = pd.DataFrame(rows).set_index("variant")
    first = list(variants)[0]
    summary["features_added_vs_baseline"] = [
        len(feature_sets[v] - feature_sets[first]) for v in summary.index]
    summary["features_lost_vs_baseline"] = [
        len(feature_sets[first] - feature_sets[v]) for v in summary.index]

    ordered = ["n_features", "features_added_vs_baseline", "features_lost_vs_baseline",
               "silhouette_pca", "knn_purity", "ari", "nmi", "umap_separation",
               "mean_abs_residual", "n_clusters", "n_cells", "n_celltypes", "model"]
    summary = summary[[c for c in ordered if c in summary.columns]]
    summary.to_csv(os.path.join(out_dir, "variant_comparison.tsv"), sep="\t")
    with open(os.path.join(out_dir, "variant_features.json"), "w") as fh:
        json.dump({k: sorted(v) for k, v in feature_sets.items()}, fh, indent=1)

    print("\n================ variant comparison ================")
    print(summary.to_string())
    print(f"\nwrote {os.path.join(out_dir, 'variant_comparison.tsv')}")
    print("Lower silhouette_pca / knn_purity / umap_separation than baseline means "
          "the embedding blurred; read them together with n_features.")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Compare model variants through module_1")
    parser.add_argument("--adata", required=True)
    parser.add_argument("--out", default="variant_comparison")
    parser.add_argument("--celltype-col", default="majority_celltype")
    parser.add_argument("--gene-column", default="gene_symbol")
    parser.add_argument("--and-strategy", default="median", choices=["min", "median", "mean"])
    parser.add_argument("--or-strategy", default="sum", choices=["sum", "max"])
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--variants", nargs="*", default=None,
                        help="subset of " + ", ".join(DEFAULT_VARIANTS))
    args = parser.parse_args()

    variants = DEFAULT_VARIANTS if not args.variants else {
        k: DEFAULT_VARIANTS[k] for k in args.variants}
    compare_variants(args.adata, args.out, variants,
                     celltype_col=args.celltype_col, gene_column=args.gene_column,
                     and_strategy=args.and_strategy, or_strategy=args.or_strategy,
                     cluster_resolution=args.resolution)


if __name__ == "__main__":
    main()
