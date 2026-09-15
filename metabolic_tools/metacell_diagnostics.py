import re
import ast
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
from .gene_mapping import resolve_model_path, map_var_names_to_model

SPECIES_PREFIX = {'hsapiens': 'ENSG', 'mmusculus': 'ENSMUSG', 'drerio': 'ENSDARG'}
AND_OPS = {'min': np.min, 'median': np.median, 'mean': np.mean}
OR_OPS = {'sum': np.sum, 'max': np.max}
COUNT_COLUMNS = ['total_counts', 'n_counts', 'nCount_RNA', 'nUMI']


# =========================================================
# 1. RAW COUNT RECOVERY
# =========================================================
def recover_counts(adata, tol=0.05, min_integer_fraction=0.99, max_divisor=3):
    """
    Reverses log1p(normalize_total) back to integer UMI counts.

    Within a cell, expm1(X) is counts * (a per-cell size factor), so the smallest
    non-zero value is the size factor for 1 UMI. Dividing by it should give whole
    numbers; this is checked rather than assumed. Cells whose smallest count is 2 or 3
    (no singleton genes) are corrected by trying small divisors.

    Writes adata.layers['counts'] and adata.obs['total_counts_recovered'] in place and
    returns a report dict. Raises ValueError if the data cannot be recovered exactly
    (scaled, regressed, batch-corrected or SCTransformed matrices).
    """
    X = adata.X
    X = X.tocsr().astype(np.float64) if sp.issparse(X) else sp.csr_matrix(np.asarray(X, dtype=np.float64))
    X.eliminate_zeros()
    if X.nnz == 0:
        raise ValueError("adata.X is empty.")
    if X.data.min() < 0:
        raise ValueError("adata.X contains negative values (scaled or regressed data). "
                         "Counts cannot be recovered; download the raw matrices from GEO instead.")

    report = {}
    n_cells = X.shape[0]
    nnz_per_cell = np.diff(X.indptr)
    row_idx = np.repeat(np.arange(n_cells), nnz_per_cell)

    already_integer = np.mean(np.abs(X.data - np.round(X.data)) < tol)
    if X.data.max() > 50 and already_integer >= min_integer_fraction:
        counts = X
        report['source'] = 'adata.X already holds counts'
    else:
        E = X.copy()
        E.data = np.expm1(E.data)

        nonempty = nnz_per_cell > 0
        size_factor = np.full(n_cells, np.nan)
        size_factor[nonempty] = np.minimum.reduceat(E.data, X.indptr[:-1][nonempty])

        # Pick, per cell, the divisor (1 UMI, 2 UMIs, ...) that makes the values most integer-like
        best_frac = np.full(n_cells, -1.0)
        best_divisor = np.ones(n_cells)
        for k in range(1, max_divisor + 1):
            vals = E.data / (size_factor[row_idx] / k)
            is_int = (np.abs(vals - np.round(vals)) < tol).astype(float)
            frac = np.bincount(row_idx, weights=is_int, minlength=n_cells) / np.maximum(nnz_per_cell, 1)
            better = frac > best_frac + 1e-9
            best_frac[better] = frac[better]
            best_divisor[better] = k

        # A near-zero CV means every cell was scaled to the same total (normalize_total over all genes)
        norm_totals = np.bincount(row_idx, weights=E.data, minlength=n_cells)
        report['normalised_total_cv'] = float(np.std(norm_totals[nonempty]) / np.mean(norm_totals[nonempty]))
        report['source'] = 'reversed log1p(normalize_total)'
        report['cells_needing_divisor>1'] = int(np.sum(best_divisor > 1))

        counts = E
        counts.data = E.data / (size_factor[row_idx] / best_divisor[row_idx])

    residual = np.abs(counts.data - np.round(counts.data))
    integer_fraction = float(np.mean(residual < tol))
    report['integer_fraction'] = integer_fraction
    per_cell = np.bincount(row_idx, weights=(residual < tol).astype(float), minlength=n_cells) / np.maximum(nnz_per_cell, 1)
    report['cells_below_99pct_integer'] = int(np.sum(per_cell[nnz_per_cell > 0] < 0.99))

    if integer_fraction < min_integer_fraction:
        raise ValueError(f"Only {integer_fraction:.1%} of recovered values are integers. adata.X is not a simple "
                         "log1p(normalize_total) transform; download the raw matrices from GEO instead.")

    counts.data = np.round(counts.data)
    counts.eliminate_zeros()
    counts = counts.astype(np.float32)
    adata.layers['counts'] = counts
    adata.obs['total_counts_recovered'] = np.asarray(counts.sum(axis=1)).ravel()

    # Cross-check against any library size column shipped with the data
    for col in COUNT_COLUMNS:
        if col in adata.obs.columns:
            report[f'pearson_r_vs_{col}'] = float(np.corrcoef(adata.obs[col].astype(float), adata.obs['total_counts_recovered'])[0, 1])

    report['median_umis_per_cell'] = float(np.median(adata.obs['total_counts_recovered']))
    return report


# =========================================================
# 2. GPR RULE PARSING (mirrors calculate_ecs)
# =========================================================
def _rule_string(rxn):
    notes = rxn.get('notes', {})
    if isinstance(notes, dict) and 'GENE_ASSOCIATION' in notes:
        return str(notes.get('GENE_ASSOCIATION', ''))
    return str(rxn.get('gene_reaction_rule', ''))


def _rule_genes(rule):
    return set(t for t in re.findall(r'[a-zA-Z0-9\-\.]+', rule) if t.lower() not in ('and', 'or'))


def _rule_category(rule, species_prefix):
    """Same species-block categorisation as cleaning_report."""
    blocks = re.split(r'\)\s*or\s*\(', rule, flags=re.IGNORECASE)
    species_rule = next((b for b in blocks if species_prefix in b), rule)
    n_genes = len({g for g in _rule_genes(species_rule) if g.startswith(species_prefix)})
    if n_genes == 0:
        return 'none'
    if n_genes == 1:
        return 'single_gene'
    has_and = ' and ' in species_rule.lower()
    has_or = ' or ' in species_rule.lower()
    if has_and and has_or:
        return 'mixed_and_or'
    return 'complex_and' if has_and else ('isozyme_or' if has_or else 'single_gene')


def _compile_rule(rule):
    genes = sorted(_rule_genes(rule))
    safe_rule, mapping = rule, {}
    for i, gene in enumerate(genes):
        safe_name = f"VAR_{i}"
        mapping[safe_name] = gene
        safe_rule = re.sub(rf'(?<![a-zA-Z0-9\-\.]){re.escape(gene)}(?![a-zA-Z0-9\-\.])', safe_name, safe_rule)
    safe_rule = re.sub(r'\bAND\b', 'and', safe_rule, flags=re.IGNORECASE)
    safe_rule = re.sub(r'\bOR\b', 'or', safe_rule, flags=re.IGNORECASE)
    try:
        return ast.parse(safe_rule, mode='eval').body, mapping
    except SyntaxError:
        return None, mapping


def _evaluate(node, mapping, values, n, and_op, or_op):
    if isinstance(node, ast.Name):
        return values.get(mapping.get(node.id, node.id), np.zeros(n))
    if isinstance(node, ast.BoolOp):
        arrays = [_evaluate(v, mapping, values, n, and_op, or_op) for v in node.values]
        return and_op(arrays, axis=0) if isinstance(node.op, ast.And) else or_op(arrays, axis=0)
    return np.zeros(n)


def _group_means(matrix, labels):
    groups = pd.Index(pd.unique(labels))
    codes = groups.get_indexer(labels)
    indicator = sp.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(len(groups), len(codes)))
    sizes = np.asarray(indicator.sum(axis=1)).ravel()
    means = sp.diags(1.0 / sizes) @ indicator @ (matrix if sp.issparse(matrix) else sp.csr_matrix(matrix))
    return groups, np.asarray(means.todense())


# =========================================================
# 3. GENE DROPOUT LEVERAGE
# =========================================================
def gene_dropout_leverage(
    adata,                        # AnnData: single cells, adata.X on the same scale calculate_ecs uses.
    groupby,                      # str: adata.obs column holding cell types.
    model_path='default',         # str: model JSON, or "default" for the packaged mitoMAMMAL model.
    species='mmusculus',          # str: 'mmusculus', 'hsapiens' or 'drerio'.
    symbol_col='gene_symbol',     # str: adata.var column with gene symbols (used if var_names are not model IDs).
    and_strategy='median',        # str: AND operator, matching calculate_ecs.
    or_strategy='sum',            # str: OR operator, matching calculate_ecs.
    split_isozymes=True           # bool: score isozyme OR branches separately, matching calculate_ecs.
):
    """
    For each model gene and cell type, measures how much reaction scores depend on that gene
    being detected: the relative drop in each reaction score when the gene alone is set to zero,
    evaluated on the cell type's mean expression and summed over reactions.

    A single-gene reaction contributes 1, an isozyme under OR=sum contributes its share of the
    total, and a subunit of a large complex under AND=median contributes close to 0.

    Returns (leverage DataFrame indexed by model gene ID, the AnnData with var_names mapped to model IDs).
    """
    if and_strategy not in AND_OPS or or_strategy not in OR_OPS:
        raise ValueError(f"and_strategy must be in {list(AND_OPS)} and or_strategy in {list(OR_OPS)}")
    and_op, or_op = AND_OPS[and_strategy], OR_OPS[or_strategy]
    species_prefix = SPECIES_PREFIX[species]

    with open(resolve_model_path(model_path), 'r', encoding='utf-8') as f:
        model_json = json.load(f)
    adata = map_var_names_to_model(adata, model_json, symbol_col, species_prefix)

    dataset_genes = {g: i for i, g in enumerate(adata.var_names.astype(str))}
    groups, means = _group_means(adata.X, adata.obs[groupby].astype(str).values)
    n = len(groups)

    # Collect scored features exactly as calculate_ecs would (deduplicated rules, split isozymes)
    features, seen = [], set()
    for rxn in model_json.get('reactions', []):
        if not isinstance(rxn, dict):
            continue
        rule = _rule_string(rxn)
        if not rule or rule == 'nan':
            continue
        present = {g for g in _rule_genes(rule) if g in dataset_genes}
        if not present:
            continue
        signature = (tuple(sorted(present)), ' and ' in rule.lower(), ' or ' in rule.lower())
        if signature in seen:
            continue
        seen.add(signature)

        category = _rule_category(rule, species_prefix)
        if split_isozymes and category == 'isozyme_or':
            for branch in re.split(r'\s+or\s+', rule, flags=re.IGNORECASE):
                branch = branch.strip('() ')
                if any(g in dataset_genes for g in _rule_genes(branch)):
                    features.append((branch, category))
        else:
            features.append((rule, category))

    leverage = {}
    n_features = {}
    category_counts = {}
    for rule, category in features:
        tree, mapping = _compile_rule(rule)
        if tree is None:
            continue
        genes = [g for g in mapping.values() if g in dataset_genes]
        values = {g: means[:, dataset_genes[g]] for g in genes}
        base = _evaluate(tree, mapping, values, n, and_op, or_op)
        with np.errstate(divide='ignore', invalid='ignore'):
            for gene in genes:
                knocked = dict(values)
                knocked[gene] = np.zeros(n)
                drop = np.where(base > 0, (base - _evaluate(tree, mapping, knocked, n, and_op, or_op)) / base, 0.0)
                leverage[gene] = leverage.get(gene, np.zeros(n)) + np.clip(drop, 0, 1)
                n_features[gene] = n_features.get(gene, 0) + 1
                category_counts.setdefault(gene, {}).setdefault(category, 0)
                category_counts[gene][category] += 1

    df = pd.DataFrame.from_dict(leverage, orient='index', columns=list(groups))
    df.index.name = 'model_gene'
    cats = pd.DataFrame.from_dict(category_counts, orient='index').fillna(0).astype(int).add_prefix('n_')
    df.insert(0, 'n_features', pd.Series(n_features))
    df = cats.join(df, how='right')
    if symbol_col in adata.var.columns:
        var = adata.var[~adata.var.index.duplicated()]
        df.insert(0, 'symbol', var[symbol_col].reindex(df.index).astype(str).values)
    else:
        df.insert(0, 'symbol', df.index)
    return df, adata


# =========================================================
# 4. DROPOUT DIAGNOSTIC
# =========================================================
def _weighted_quantile(values, weights, q):
    ok = np.isfinite(values) & (weights > 0)
    if not ok.any():
        return np.nan
    order = np.argsort(values[ok])
    v, w = values[ok][order], weights[ok][order]
    return float(v[np.searchsorted(np.cumsum(w), q * w.sum())])


def aggregate_counts(adata, label_col, groupby, counts_layer='counts'):
    """Sums counts over metacell labels (e.g. SEACells output) so they can go through dropout_diagnostic."""
    labels = adata.obs[label_col].astype(str).values
    groups = pd.Index(pd.unique(labels))
    codes = groups.get_indexer(labels)
    indicator = sp.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(len(groups), len(codes)))
    summed = (indicator @ sp.csr_matrix(adata.layers[counts_layer])).tocsr()
    obs = pd.DataFrame(index=groups)
    obs[groupby] = adata.obs.groupby(label_col, observed=True)[groupby].agg(lambda s: s.value_counts().index[0]).reindex(groups).astype(str).values
    obs['n_cells'] = np.asarray(indicator.sum(axis=1)).ravel()
    out = ad.AnnData(X=summed, obs=obs, var=adata.var.copy())
    out.layers['counts'] = summed
    return out


def dropout_diagnostic(
    adata,                        # AnnData: cells or metacells with a raw counts layer and model gene var_names.
    leverage,                     # DataFrame: output of gene_dropout_leverage.
    groupby,                      # str: adata.obs column holding cell types.
    counts_layer='counts',        # str: layer with raw integer counts.
    detection_prob=0.95,          # float: target probability of detecting a gene.
    reference_size=50             # int: metacell size to benchmark against (e.g. SEACells target_metacell_size).
):
    """
    Per gene and cell type, compares observed zeros to the zeros expected from sequencing depth
    alone (Poisson: P(zero) = exp(-mu_g * library_size)), and estimates how many median-depth
    units must be pooled to detect the gene with probability `detection_prob`.

    Observed zeros close to expected means dropout is sampling-driven and pooling will fix it.
    Observed zeros well above expected means the gene is genuinely on in some cells and off in
    others, so pooling would blur real heterogeneity.

    Returns (per-gene DataFrame, per-cell-type summary DataFrame), with summaries weighted by leverage.
    """
    counts = sp.csr_matrix(adata.layers[counts_layer])
    library = np.asarray(counts.sum(axis=1)).ravel()
    genes = [g for g in leverage.index if g in adata.var_names]
    gene_idx = adata.var_names.get_indexer(genes)
    labels = adata.obs[groupby].astype(str).values

    rows, summary = [], []
    for group in pd.unique(labels):
        if group not in leverage.columns:
            continue
        mask = labels == group
        sub = counts[mask][:, gene_idx]
        lib = library[mask]
        n_units = int(mask.sum())

        zero_frac = 1 - sub.getnnz(axis=0) / n_units
        mu = np.asarray(sub.sum(axis=0)).ravel() / lib.sum()
        expected_zero = sum(np.exp(-np.outer(lib[i:i + 2000], mu)).sum(axis=0) for i in range(0, n_units, 2000)) / n_units
        with np.errstate(divide='ignore'):
            units_needed = np.where(mu > 0, -np.log(1 - detection_prob) / (mu * np.median(lib)), np.inf)

        weight = leverage.loc[genes, group].to_numpy(dtype=float)
        rows.append(pd.DataFrame({
            'group': group, 'model_gene': genes, 'symbol': leverage.loc[genes, 'symbol'].values,
            'leverage': weight, 'zero_frac': zero_frac, 'expected_zero_frac': expected_zero,
            'excess_zero_frac': zero_frac - expected_zero, 'units_needed': units_needed,
        }))

        w_total = weight.sum()
        detected = np.isfinite(units_needed)
        summary.append({
            'group': group,
            'n_units': n_units,
            'median_library_size': float(np.median(lib)),
            'weighted_zero_frac': float(np.dot(weight, zero_frac) / w_total),
            'weighted_expected_zero_frac': float(np.dot(weight, expected_zero) / w_total),
            'weighted_excess_zero_frac': float(np.dot(weight, zero_frac - expected_zero) / w_total),
            'leverage_never_detected': float(weight[~detected].sum() / w_total),
            'units_needed_weighted_median': _weighted_quantile(units_needed, weight, 0.5),
            'units_needed_weighted_p90': _weighted_quantile(units_needed, weight, 0.9),
            f'leverage_covered_at_{reference_size}': float(weight[units_needed <= reference_size].sum() / w_total),
        })

    return pd.concat(rows, ignore_index=True), pd.DataFrame(summary).set_index('group')
