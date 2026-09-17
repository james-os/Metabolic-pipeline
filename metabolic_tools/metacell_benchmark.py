import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
from scipy.stats import spearmanr
from .gene_mapping import resolve_model_path
from .metacell_diagnostics import SPECIES_PREFIX, AND_OPS, OR_OPS, _collect_features, _compile_rule, _evaluate
from .metabolic_metacells import _balanced_partition, compute_pca, aggregate_metacells, metacell_detection


# =========================================================
# 1. SIMULATING SHALLOWER SEQUENCING
# =========================================================
def thin_counts(adata, fraction, counts_layer='counts', random_state=0, target_sum=1e4):
    """
    Keeps each UMI independently with probability `fraction` (binomial thinning), mimicking the same
    cells sequenced more shallowly. Returns a new AnnData with thinned counts in layers['counts'] and
    X re-normalised to log1p(counts per 10k). The embedding is not copied, so it is recomputed from
    the thinned data, as it would be in a real shallower experiment.
    """
    rng = np.random.default_rng(random_state)
    counts = sp.csr_matrix(adata.layers[counts_layer], dtype=np.float64, copy=True)
    counts.data = rng.binomial(np.round(counts.data).astype(np.int64), fraction).astype(np.float64)
    counts.eliminate_zeros()
    total = np.asarray(counts.sum(axis=1)).ravel()
    norm = sp.csr_matrix(sp.diags(target_sum / np.maximum(total, 1)) @ counts)
    norm.data = np.log1p(norm.data)
    out = ad.AnnData(X=norm.astype(np.float32), obs=adata.obs.copy(), var=adata.var.copy())
    out.layers['counts'] = counts.astype(np.float32)
    return out


# =========================================================
# 2. COMPARISON GROUPINGS
# =========================================================
def fixed_size_labels(adata, celltype_col, cells_per_metacell=50, use_rep='X_pca', n_pcs=30, random_state=0):
    """
    SEACells-style sizing: within each cell type, a fixed number of cells per metacell (an int, or a
    dict per cell type), grouped by similarity in the embedding, with samples mixed.
    """
    if use_rep not in adata.obsm:
        compute_pca(adata, n_pcs=n_pcs, key=use_rep)
    coords = np.asarray(adata.obsm[use_rep])[:, :n_pcs]
    labels = pd.Series(index=adata.obs_names, dtype=object)
    for celltype, idx in adata.obs.groupby(adata.obs[celltype_col].astype(str)).indices.items():
        size = cells_per_metacell[celltype] if isinstance(cells_per_metacell, dict) else cells_per_metacell
        n_groups = int(max(1, round(len(idx) / size)))
        part = _balanced_partition(coords[idx], np.ones(len(idx)), n_groups, random_state=random_state)
        labels.iloc[idx] = [f'{celltype}|fixed|{k}' for k in part]
    return labels


def seacells_labels(adata, celltype_col, cells_per_metacell=50, use_rep='X_pca', n_pcs=30, max_iter=50):
    """
    SEACells metacells within each cell type (samples mixed), as in stratify_metacells.
    Requires SEACells to be installed. Cell types SEACells cannot handle become one metacell.
    """
    import SEACells

    if not hasattr(ad.AnnData, '_is_patched_for_seacells'):
        _original_init = ad.AnnData.__init__

        def _patched_init(self, *args, **kwargs):
            kwargs.pop('dtype', None)
            _original_init(self, *args, **kwargs)

        ad.AnnData.__init__ = _patched_init
        ad.AnnData._is_patched_for_seacells = True

    if use_rep not in adata.obsm:
        compute_pca(adata, n_pcs=n_pcs, key=use_rep)
    coords = np.asarray(adata.obsm[use_rep])[:, :n_pcs]
    labels = pd.Series(index=adata.obs_names, dtype=object)
    for celltype, idx in adata.obs.groupby(adata.obs[celltype_col].astype(str)).indices.items():
        n_groups = int(max(1, round(len(idx) / cells_per_metacell)))
        if n_groups <= 1 or len(idx) < 10:
            labels.iloc[idx] = f'{celltype}|seacells|0'
            continue
        sub = ad.AnnData(X=adata.X[idx], obs=pd.DataFrame(index=adata.obs_names[idx]))
        sub.obsm[use_rep] = coords[idx]
        try:
            model = SEACells.core.SEACells(sub, build_kernel_on=use_rep, n_SEACells=n_groups,
                                           n_waypoint_eigs=min(10, len(idx) - 1), convergence_epsilon=1e-4)
            model.construct_kernel_matrix()
            model.initialize_archetypes()
            model.fit(min_iter=10, max_iter=max_iter)
            labels.iloc[idx] = (f'{celltype}|seacells|' + sub.obs['SEACell'].astype(str)).values
        except Exception as e:
            print(f'   ! SEACells failed for {celltype} ({e}); using one metacell.')
            labels.iloc[idx] = f'{celltype}|seacells|0'
    return labels


# =========================================================
# 3. SCORING
# =========================================================
def feature_scores(mc, model_path='default', species='mmusculus', and_strategy='median', or_strategy='sum',
                   split_isozymes=True):
    """Reaction features (as calculate_ecs defines them) evaluated on each metacell's log-normalised X."""
    with open(resolve_model_path(model_path), 'r', encoding='utf-8') as f:
        model_json = json.load(f)
    dataset_genes = {g: i for i, g in enumerate(mc.var_names.astype(str))}
    X = mc.X.toarray() if sp.issparse(mc.X) else np.asarray(mc.X)
    and_op, or_op = AND_OPS[and_strategy], OR_OPS[or_strategy]
    scores = {}
    for rule, _, _, label in _collect_features(model_json, dataset_genes, SPECIES_PREFIX[species], split_isozymes):
        tree, mapping = _compile_rule(rule)
        if tree is None:
            continue
        values = {g: X[:, dataset_genes[g]] for g in mapping.values() if g in dataset_genes}
        scores[label] = _evaluate(tree, mapping, values, mc.n_obs, and_op, or_op)
    return pd.DataFrame(scores, index=mc.obs_names)


def compare_to_truth(scores_thin, scores_truth):
    """
    Per metacell, compares feature scores from thinned counts with the same metacell at full depth:
      - false_zero_rate: share of features present at full depth that read zero after thinning;
      - median_rel_error: typical relative error of features present at full depth;
      - spearman: rank agreement across all features.
    """
    rows = []
    for name in scores_thin.index:
        truth = scores_truth.loc[name].to_numpy(dtype=float)
        thin = scores_thin.loc[name].to_numpy(dtype=float)
        present = truth > 0
        rows.append({
            'metacell': name,
            'features_present': int(present.sum()),
            'false_zero_rate': float(np.mean(thin[present] == 0)) if present.any() else np.nan,
            'median_rel_error': float(np.median(np.abs(thin[present] - truth[present]) / truth[present])) if present.any() else np.nan,
            'spearman': float(spearmanr(truth, thin).correlation) if present.sum() > 2 else np.nan,
        })
    return pd.DataFrame(rows)


def compactness(adata, labels, celltype_col, use_rep='X_pca', n_pcs=30):
    """
    Mean distance of a metacell's cells to their centroid, divided by the same measure for the whole
    cell type. 0 = identical cells; 1 = as spread out as the cell type; lower is tighter.
    """
    coords = np.asarray(adata.obsm[use_rep])[:, :n_pcs]
    labels = pd.Series(labels, index=adata.obs_names).astype(str)
    celltypes = adata.obs[celltype_col].astype(str)
    spread = {}
    for celltype, idx in celltypes.groupby(celltypes.values).indices.items():
        spread[celltype] = np.linalg.norm(coords[idx] - coords[idx].mean(axis=0), axis=1).mean()
    out = {}
    for name, idx in labels.groupby(labels.values).indices.items():
        celltype = celltypes.iloc[idx].mode().iloc[0]
        within = np.linalg.norm(coords[idx] - coords[idx].mean(axis=0), axis=1).mean()
        out[name] = within / spread[celltype] if spread[celltype] > 0 else np.nan
    return pd.Series(out, name='compactness')


def evaluate_grouping(original, thinned, labels, method, fraction, celltype_col, sample_col, umi_budgets,
                      genes_df, gene_classes, model_path='default', species='mmusculus', and_strategy='median',
                      or_strategy='sum', use_rep='X_pca', budget_tolerance=0.9):
    """
    Scores one grouping of the thinned cells: dropout error against the same groups at full depth,
    gene detection, UMI budget attainment, sample mixing and compactness. One row per metacell.
    """
    mc_thin = aggregate_metacells(thinned, labels, celltype_col, sample_col)
    mc_truth = aggregate_metacells(original, labels, celltype_col, sample_col)
    kwargs = dict(model_path=model_path, species=species, and_strategy=and_strategy, or_strategy=or_strategy)
    result = compare_to_truth(feature_scores(mc_thin, **kwargs), feature_scores(mc_truth, **kwargs)).set_index('metacell')
    result = result.join(mc_thin.obs)
    result['umi_budget'] = result[celltype_col].map(umi_budgets).astype(float)
    result['budget_ratio'] = result['total_umis'] / result['umi_budget']
    result['under_budget'] = result['budget_ratio'] < budget_tolerance
    result = result.join(metacell_detection(thinned, labels, genes_df, gene_classes, celltype_col).set_index('metacell'))
    result = result.join(compactness(thinned, labels, celltype_col, use_rep))
    result.insert(0, 'fraction', fraction)
    result.insert(0, 'method', method)
    return result.reset_index()
