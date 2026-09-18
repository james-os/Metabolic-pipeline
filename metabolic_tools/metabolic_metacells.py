import json
from collections import Counter
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
from .gene_mapping import resolve_model_path
from .metacell_diagnostics import (
    SPECIES_PREFIX, AND_OPS, OR_OPS, recover_counts, gene_dropout_leverage, dropout_diagnostic,
    metacell_size_targets, _rule_string, _rule_genes, _compile_rule, _evaluate)


# =========================================================
# 1. BUILDING BLOCKS
# =========================================================
def compute_pca(adata, n_top_genes=2000, n_pcs=30, key='X_pca'):
    """PCA on the highly variable genes of the log-normalised adata.X, stored in adata.obsm[key]."""
    tmp = ad.AnnData(X=adata.X.copy(), obs=pd.DataFrame(index=adata.obs_names), var=pd.DataFrame(index=adata.var_names))
    sc.pp.highly_variable_genes(tmp, n_top_genes=min(n_top_genes, tmp.n_vars), flavor='seurat')
    tmp = tmp[:, tmp.var['highly_variable'].to_numpy()].copy()
    sc.pp.pca(tmp, n_comps=min(n_pcs, tmp.n_obs - 1, tmp.n_vars - 1))
    adata.obsm[key] = tmp.obsm['X_pca']
    return adata.obsm[key]


def _fill_starved(coords, weights, labels, centres, floor):
    """
    Tops up any group left below `floor` total weight from the groups that have weight to spare.

    The capacity cap only limits how heavy a group may become, so a group that few cells prefer can
    be left starved while every other group sits just under the cap. The caller only asks for as
    many groups as the weight can afford, so each group can reach the floor; this moves whichever
    spare cell lies closest to the starved group's centroid, which keeps the groups compact.

    Groups can still finish below the floor if no single cell can be moved without pushing its own
    group under, so callers must keep treating the floor as a target rather than a guarantee.
    """
    if floor is None:
        return labels
    n_groups = len(centres)
    load = np.bincount(labels, weights=weights, minlength=n_groups)
    labels = labels.copy()
    for _ in range(len(labels)):
        starved = np.flatnonzero(load < floor)
        if not len(starved):
            break
        g = int(starved[np.argmin(load[starved])])
        members = labels == g
        centre = np.average(coords[members], axis=0, weights=weights[members]) if members.any() else centres[g]
        spare = np.flatnonzero((labels != g) & (load[labels] - weights >= floor))
        if not len(spare):
            break
        i = int(spare[np.argmin(((coords[spare] - centre) ** 2).sum(axis=1))])
        load[labels[i]] -= weights[i]
        load[g] += weights[i]
        labels[i] = g
    return labels


def _balanced_partition(coords, weights, n_groups, n_iter=15, slack=0.1, floor=None, random_state=0):
    """
    Splits points into `n_groups` compact groups with roughly equal total weight (UMIs).

    Capacity-constrained k-means: each round, cells are assigned to the nearest centroid that still
    has room (capacity = mean group weight * (1 + slack)), handling the most clear-cut cells first
    so ambiguous cells take whatever room is left. Centroids are then UMI-weighted means.

    `floor` sets a minimum total weight per group, topped up after clustering; without it a group
    the other groups can absorb between them is left far lighter than the rest.
    """
    n = len(coords)
    if n_groups <= 1 or n <= 1:
        return np.zeros(n, dtype=int)
    n_groups = min(n_groups, n)
    rng = np.random.default_rng(random_state)

    # k-means++ seeding
    centres = [coords[rng.integers(n)]]
    for _ in range(1, n_groups):
        d2 = np.min(((coords[:, None, :] - np.asarray(centres)[None]) ** 2).sum(-1), axis=1)
        probs = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1 / n)
        centres.append(coords[rng.choice(n, p=probs)])
    centres = np.asarray(centres, dtype=float)

    capacity = weights.sum() / n_groups * (1 + slack)
    labels = np.full(n, -1)
    for _ in range(n_iter):
        dist = ((coords[:, None, :] - centres[None]) ** 2).sum(-1)
        ranked = np.sort(dist, axis=1)
        regret = ranked[:, 1] - ranked[:, 0]
        load = np.zeros(n_groups)
        new = np.empty(n, dtype=int)
        for i in np.argsort(-regret):
            prefs = np.argsort(dist[i])
            room = load[prefs] + weights[i] <= capacity
            g = prefs[np.argmax(room)] if room.any() else int(np.argmin(load))
            new[i] = g
            load[g] += weights[i]
        for g in range(n_groups):
            members = new == g
            centres[g] = np.average(coords[members], axis=0, weights=weights[members]) if members.any() else coords[rng.integers(n)]
        if np.array_equal(new, labels):
            break
        labels = new
    return _fill_starved(coords, weights, labels, centres, floor)


def aggregate_metacells(adata, labels, celltype_col, sample_col=None, counts_layer='counts', target_sum=1e4):
    """
    Sums raw counts over metacell labels. X is log1p(counts per 10k), the same scale as the single
    cells, so calculate_ecs can be run on the result; summed counts are kept in layers['counts'].
    obs records the majority cell type and sample, their purity, cell number and total UMIs.
    """
    labels = pd.Series(labels, index=adata.obs_names).astype(str)
    names = pd.Index(pd.unique(labels.values), name='metacell')
    codes = names.get_indexer(labels.values)
    indicator = sp.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(len(names), len(codes)))
    counts = (indicator @ sp.csr_matrix(adata.layers[counts_layer])).tocsr()
    total = np.asarray(counts.sum(axis=1)).ravel()

    obs = pd.DataFrame(index=names)
    for col, purity_col in ((celltype_col, 'celltype_purity'), (sample_col, 'sample_purity')):
        if col is None:
            continue
        values = adata.obs[col].astype(str).groupby(labels.values)
        obs[col] = values.agg(lambda s: s.value_counts().index[0]).reindex(names).values
        obs[purity_col] = values.agg(lambda s: s.value_counts().iloc[0] / len(s)).reindex(names).values
    obs['n_cells'] = np.asarray(indicator.sum(axis=1)).ravel().astype(int)
    obs['total_umis'] = total

    norm = sp.csr_matrix(sp.diags(target_sum / np.maximum(total, 1)) @ counts)
    norm.data = np.log1p(norm.data)
    mc = ad.AnnData(X=norm.astype(np.float32), obs=obs, var=adata.var.copy())
    mc.layers['counts'] = counts.astype(np.float32)
    return mc


# =========================================================
# 2. GENE AND REACTION CLASSES
# =========================================================
def classify_genes(adata, genes_df, targets, counts_layer='counts'):
    """
    Per cell type, puts each model gene in one of three classes:
      - 'off': no counts anywhere in the dataset (e.g. liver enzymes in cochlea);
      - 'uncertain': detected somewhere, but needs more cells than the cell type's cap to detect;
      - 'pooled': detectable within the metacell size, so metacell expression can be trusted.
    """
    dataset_total = pd.Series(np.asarray(sp.csr_matrix(adata.layers[counts_layer]).sum(axis=0)).ravel(), index=adata.var_names)
    dataset_total = dataset_total[~dataset_total.index.duplicated()]
    df = genes_df[['group', 'model_gene', 'symbol', 'leverage', 'units_needed', 'rate', 'dispersion']].copy()
    df['cap_cells'] = df['group'].map(targets['cap_cells'])
    absent = dataset_total.reindex(df['model_gene']).fillna(0).to_numpy() == 0
    df['class'] = np.select([absent, df['units_needed'] > df['cap_cells']], ['off', 'uncertain'], 'pooled')
    return df


def classify_reactions(model_json, gene_classes, dataset_genes, species, and_strategy='median', or_strategy='sum'):
    """
    Per cell type, classifies each reaction from its genes' classes using the same AND/OR rules as
    calculate_ecs:
      - 'pooled' if the pooled genes alone can support the reaction (transcriptomics constrains its bounds);
      - 'uncertain' if it needs uncertain or unmeasured genes (keep default model bounds, flagged);
      - 'off' if only genes absent from the whole dataset could support it (close; reopen if that
        breaks feasibility or an essential flux).
    Genes in the model but not measured in the dataset count as uncertain, not off.
    """
    and_op, or_op = AND_OPS[and_strategy], OR_OPS[or_strategy]
    prefix = SPECIES_PREFIX[species]
    lookup = {g: dict(zip(d['model_gene'], d['class'])) for g, d in gene_classes.groupby('group')}
    one, zero = np.ones(1), np.zeros(1)

    rows = []
    for rxn in model_json.get('reactions', []):
        if not isinstance(rxn, dict):
            continue
        rule = _rule_string(rxn)
        if not rule or rule == 'nan':
            continue
        genes = sorted(g for g in _rule_genes(rule) if g.startswith(prefix))
        if not genes:
            continue
        tree, mapping = _compile_rule(rule)
        if tree is None:
            continue
        rxn_id = str(rxn.get('id', 'Unknown_Reaction'))
        for group, classes in lookup.items():
            c = {g: classes.get(g, 'uncertain') if g in dataset_genes else 'uncertain' for g in genes}
            pooled_only = {g: one if k == 'pooled' else zero for g, k in c.items()}
            possible = {g: one if k != 'off' else zero for g, k in c.items()}
            if _evaluate(tree, mapping, pooled_only, 1, and_op, or_op)[0] > 0:
                klass = 'pooled'
            elif _evaluate(tree, mapping, possible, 1, and_op, or_op)[0] > 0:
                klass = 'uncertain'
            else:
                klass = 'off'
            counts = Counter(c.values())
            rows.append({'group': group, 'reaction_id': rxn_id, 'class': klass,
                         'n_pooled_genes': counts['pooled'],
                         'n_uncertain_genes': counts['uncertain'],
                         'n_off_genes': counts['off']})
    return pd.DataFrame(rows)


def metacell_detection(adata, labels, genes_df, gene_classes, celltype_col, counts_layer='counts'):
    """
    Per metacell, the leverage-weighted share of genes detected (at least one UMI), both for the
    'pooled' genes the size target aims to cover and for all model genes, plus the detection the
    dropout model predicted for the pooled genes. Comparing predicted and observed checks the model.
    """
    counts = sp.csr_matrix(adata.layers[counts_layer])
    library = np.asarray(counts.sum(axis=1)).ravel().astype(float)
    labels = pd.Series(labels, index=adata.obs_names).astype(str)
    celltypes = adata.obs[celltype_col].astype(str).to_numpy()

    per_type = {}
    for group, g in genes_df[genes_df['leverage'] > 0].groupby('group'):
        g = g.set_index('model_gene')
        g = g[g.index.isin(adata.var_names)]
        pooled = set(gene_classes.loc[(gene_classes['group'] == group) & (gene_classes['class'] == 'pooled'), 'model_gene'])
        per_type[group] = (g, adata.var_names.get_indexer(g.index), g.index.isin(pooled))

    rows = []
    for name, idx in labels.groupby(labels.values).indices.items():
        group = pd.Series(celltypes[idx]).mode().iloc[0]
        g, gene_idx, is_pooled = per_type[group]
        detected = np.asarray(counts[idx][:, gene_idx].sum(axis=0)).ravel() > 0
        w = g['leverage'].to_numpy(float)
        disp = np.maximum(g['dispersion'].to_numpy(float), 1e-8)
        log_p0 = -(np.log1p(np.outer(library[idx], g['rate'].to_numpy(float) * disp)) / disp).sum(axis=0)
        expected = 1 - np.exp(log_p0)
        wp = w * is_pooled
        rows.append({
            'metacell': name,
            'detected_leverage_pooled': float(wp[detected].sum() / wp.sum()) if wp.sum() else np.nan,
            'expected_detected_leverage_pooled': float((wp * expected).sum() / wp.sum()) if wp.sum() else np.nan,
            'detected_leverage_all': float(w[detected].sum() / w.sum()) if w.sum() else np.nan,
        })
    return pd.DataFrame(rows)


# =========================================================
# 3. METABOLIC METACELLS
# =========================================================
def metabolic_metacells(
    adata,                        # AnnData: single cells, log-normalised X; raw counts in counts_layer or recoverable.
    celltype_col,                 # str: adata.obs column with cell types. Metacells never mix cell types.
    sample_col,                   # str: adata.obs column with samples. Metacells never mix samples.
    model_path='default',         # str: model JSON, or "default" for the packaged mitoMAMMAL model.
    species='mmusculus',          # str: 'mmusculus', 'hsapiens' or 'drerio'.
    symbol_col='gene_symbol',     # str: adata.var column with gene symbols.
    and_strategy='median',        # str: AND operator, matching calculate_ecs.
    or_strategy='sum',            # str: OR operator, matching calculate_ecs.
    split_isozymes=True,          # bool: size for each isozyme branch separately (as calculate_ecs scores them)
                                  #       rather than for the summed reaction; False weights an isozyme by its
                                  #       share of the sum, which is what a flux bound actually depends on.
    sizing_and_strategy=None,     # str: AND operator used to weight genes when sizing, if it should differ from
                                  #      the one used to score. None follows and_strategy. 'min' sizes for every
                                  #      complex subunit while and_strategy='median' keeps scoring robust to the
                                  #      dropout that remains.
    counts_layer='counts',        # str: raw counts layer; recovered from adata.X if missing.
    coverage=0.8,                 # float: share of reachable leverage each metacell should detect.
    min_metacells=3,              # int: fewest metacells per cell type (sets the size cap).
    detection_prob=0.95,          # float: detection probability used for size targets.
    null_model='nb',              # str: 'nb' or 'poisson' dropout model.
    use_rep='X_pca',              # str: adata.obsm embedding used to group similar cells; computed if missing.
    n_top_genes=2000,             # int: highly variable genes for the PCA, if computed.
    n_pcs=30,                     # int: principal components used.
    budget_tolerance=0.9,         # float: metacells below this fraction of their UMI budget are flagged under budget.
    slack=0.1,                    # float: how far above the mean UMI load a metacell may go during partitioning.
    random_state=0                # int: seed for the partitioning.
):
    """
    Builds metacells sized to protect detection of reaction-relevant genes.

    1. Scores each gene's importance to reaction scores (whole-dataset leverage) and fits the
       dropout model per cell type. `sizing_and_strategy` and `split_isozymes` control that
       weighting alone; `and_strategy` and `or_strategy` stay the operators the scores are read
       with, so metacells can be sized conservatively and still be scored robustly.
    2. Sets a UMI budget per cell type: the pool needed to detect `coverage` of reachable leverage,
       capped so every cell type keeps at least `min_metacells` metacells.
    3. Within each cell type x sample, splits cells into as many groups as the stratum's UMIs can
       keep at the budget (at most floor(total UMIs / budget)), compact in the embedding and
       balanced in UMIs. Whole cells cannot always be split that finely, so a split that still
       leaves a metacell short is retried with one group fewer: fewer, fuller metacells rather than
       a starved one. A cell type x sample with fewer UMIs than the budget becomes a single metacell
       flagged 'under_budget', so that flag marks a stratum too small to afford a metacell rather
       than an uneven split.
    4. Classifies genes and reactions per cell type as pooled / uncertain / off.

    Returns (metacell AnnData, info dict with 'labels', 'targets', 'gene_classes',
    'reaction_classes', 'leverage', 'genes' and 'cells', the cell-level AnnData used).
    """
    if counts_layer not in adata.layers:
        recover_counts(adata)
        counts_layer = 'counts'

    sizing_and = and_strategy if sizing_and_strategy is None else sizing_and_strategy
    leverage, adata = gene_dropout_leverage(adata, celltype_col, model_path=model_path, species=species,
                                            symbol_col=symbol_col, and_strategy=sizing_and,
                                            or_strategy=or_strategy, split_isozymes=split_isozymes,
                                            reference='global')
    genes_df, summary = dropout_diagnostic(adata, leverage, celltype_col, counts_layer=counts_layer,
                                           detection_prob=detection_prob, model=null_model)
    targets, _ = metacell_size_targets(genes_df, summary, coverage=coverage, min_metacells=min_metacells)

    if use_rep not in adata.obsm:
        compute_pca(adata, n_top_genes=n_top_genes, n_pcs=n_pcs, key=use_rep)
    coords = np.asarray(adata.obsm[use_rep])[:, :n_pcs]
    library = np.asarray(sp.csr_matrix(adata.layers[counts_layer]).sum(axis=1)).ravel().astype(float)

    labels = pd.Series(index=adata.obs_names, dtype=object)
    strata = adata.obs[[celltype_col, sample_col]].astype(str)
    for (celltype, sample), idx in strata.groupby([celltype_col, sample_col]).indices.items():
        budget = float(targets.loc[celltype, 'target_umis'])
        n_groups = int(max(1, min(len(idx), library[idx].sum() // budget)))
        while True:
            part = _balanced_partition(coords[idx], library[idx], n_groups, slack=slack, floor=budget,
                                       random_state=random_state)
            load = np.bincount(part, weights=library[idx], minlength=n_groups)
            if n_groups == 1 or load.min() >= budget:
                break
            n_groups -= 1
        labels.iloc[idx] = [f'{celltype}|{sample}|{k}' for k in part]

    mc = aggregate_metacells(adata, labels, celltype_col, sample_col, counts_layer)
    mc.obs['umi_budget'] = mc.obs[celltype_col].map(targets['target_umis']).astype(float)
    mc.obs['budget_ratio'] = mc.obs['total_umis'] / mc.obs['umi_budget']
    mc.obs['under_budget'] = mc.obs['budget_ratio'] < budget_tolerance

    with open(resolve_model_path(model_path), 'r', encoding='utf-8') as f:
        model_json = json.load(f)
    gene_classes = classify_genes(adata, genes_df, targets, counts_layer)
    reaction_classes = classify_reactions(model_json, gene_classes, set(adata.var_names.astype(str)), species,
                                          and_strategy, or_strategy)

    detection = metacell_detection(adata, labels, genes_df, gene_classes, celltype_col, counts_layer).set_index('metacell')
    for col in detection.columns:
        mc.obs[col] = detection[col].reindex(mc.obs_names).values

    info = {'labels': labels, 'targets': targets, 'gene_classes': gene_classes,
            'reaction_classes': reaction_classes, 'leverage': leverage, 'genes': genes_df, 'cells': adata}
    return mc, info
