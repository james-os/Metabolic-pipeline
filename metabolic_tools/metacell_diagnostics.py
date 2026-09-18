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


def _collect_features(model_json, dataset_genes, species_prefix, split_isozymes=True):
    """
    The scored features calculate_ecs would produce: one per deduplicated rule, or one per isozyme
    branch when split_isozymes is on. Returns a list of (rule, category, reaction_id, feature_label).
    """
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
        rxn_id = str(rxn.get('id', 'Unknown_Reaction'))
        if split_isozymes and category == 'isozyme_or':
            for i, branch in enumerate(re.split(r'\s+or\s+', rule, flags=re.IGNORECASE)):
                branch = branch.strip('() ')
                if any(g in dataset_genes for g in _rule_genes(branch)):
                    features.append((branch, category, rxn_id, f'{rxn_id}_iso{i + 1}'))
        else:
            features.append((rule, category, rxn_id, rxn_id))
    return features


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
    split_isozymes=True,          # bool: score isozyme OR branches separately, matching calculate_ecs.
    reference='group'             # str: 'group' evaluates rules on each cell type's mean expression;
                                  #      'global' uses the whole-dataset mean for every cell type.
):
    """
    For each model gene and cell type, measures how much reaction scores depend on that gene
    being detected: the relative drop in each reaction score when the gene alone is set to zero,
    evaluated on mean expression and summed over reactions.

    A single-gene reaction contributes 1, an isozyme under OR=sum contributes its share of the
    total, and a subunit of a large complex under AND=median contributes close to 0.

    With reference='group', a gene absent from a cell type gets zero leverage there. With
    reference='global', leverage reflects the gene's role in the model as expressed across the
    whole dataset, so genes missing from a cell type stay visible in dropout_diagnostic.

    Returns (leverage DataFrame indexed by model gene ID, the AnnData with var_names mapped to model IDs).
    The DataFrame's 'reactions' column lists the reaction IDs each gene feeds into.
    """
    if and_strategy not in AND_OPS or or_strategy not in OR_OPS:
        raise ValueError(f"and_strategy must be in {list(AND_OPS)} and or_strategy in {list(OR_OPS)}")
    if reference not in ('group', 'global'):
        raise ValueError(f"reference must be 'group' or 'global', got '{reference}'")
    and_op, or_op = AND_OPS[and_strategy], OR_OPS[or_strategy]
    species_prefix = SPECIES_PREFIX[species]

    with open(resolve_model_path(model_path), 'r', encoding='utf-8') as f:
        model_json = json.load(f)
    adata = map_var_names_to_model(adata, model_json, symbol_col, species_prefix)

    dataset_genes = {g: i for i, g in enumerate(adata.var_names.astype(str))}
    groups, means = _group_means(adata.X, adata.obs[groupby].astype(str).values)
    if reference == 'global':
        overall = np.asarray(adata.X.mean(axis=0)).ravel()
        means = np.tile(overall, (len(groups), 1))
    n = len(groups)

    features = _collect_features(model_json, dataset_genes, species_prefix, split_isozymes)

    leverage = {}
    n_features = {}
    category_counts = {}
    reactions = {}
    for rule, category, rxn_id, _ in features:
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
                reactions.setdefault(gene, set()).add(rxn_id)

    df = pd.DataFrame.from_dict(leverage, orient='index', columns=list(groups))
    df.index.name = 'model_gene'
    cats = pd.DataFrame.from_dict(category_counts, orient='index').fillna(0).astype(int).add_prefix('n_')
    df.insert(0, 'reactions', pd.Series({g: ';'.join(sorted(r)) for g, r in reactions.items()}))
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


def _nb_dispersion(counts, library, min_total=20, n_bins=30):
    """
    Negative binomial dispersion for every gene in one cell type, with counts modelled as
    mean = rate_g * library_size_i and variance = mean + dispersion * mean^2.

    Returns (per-gene method-of-moments dispersion, dispersion trend). The trend is the running
    median of per-gene dispersion against mean expression, i.e. how much extra variability is
    typical for genes at that expression level. Using the trend rather than each gene's own
    dispersion is what lets unusually uneven genes stand out instead of explaining themselves away.
    """
    counts = sp.csr_matrix(counts)
    total = np.asarray(counts.sum(axis=0)).ravel()
    lib_sq = float(np.sum(library ** 2))
    rate = total / library.sum()
    sum_sq = np.asarray(counts.multiply(counts).sum(axis=0)).ravel()
    sum_y_lib = np.asarray(counts.T @ library).ravel()
    residual_ss = sum_sq - 2 * rate * sum_y_lib + rate ** 2 * lib_sq
    with np.errstate(divide='ignore', invalid='ignore'):
        dispersion = np.where(total > 0, (residual_ss - total) / (rate ** 2 * lib_sq), np.nan)
    dispersion = np.clip(dispersion, 0, None)

    mean_count = total / len(library)
    fit = (total >= min_total) & np.isfinite(dispersion)
    if fit.sum() < 3 * n_bins:
        trend_value = float(np.median(dispersion[fit])) if fit.any() else 0.0
        return dispersion, np.full(len(total), trend_value)

    x = np.log10(mean_count[fit])
    y = dispersion[fit]
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    bin_of = np.clip(np.searchsorted(edges, x, side='right') - 1, 0, n_bins - 1)
    occupied = [b for b in range(n_bins) if np.any(bin_of == b)]
    centres = np.array([np.median(x[bin_of == b]) for b in occupied])
    medians = np.array([np.median(y[bin_of == b]) for b in occupied])
    trend = np.interp(np.log10(np.maximum(mean_count, 1e-12)), centres, medians)
    return dispersion, np.maximum(trend, 0)


def _p_zero(mean, dispersion):
    """P(count = 0) under NB(mean, dispersion); reduces to Poisson exp(-mean) as dispersion -> 0."""
    dispersion = np.maximum(dispersion, 1e-8)
    return np.exp(-np.log1p(dispersion * mean) / dispersion)


def nb_dispersion(adata, groupby, group, counts_layer='counts', symbol_col='gene_symbol'):
    """Per-gene dispersion and the expression-matched dispersion trend for one cell type, for plotting and checks."""
    mask = (adata.obs[groupby].astype(str) == str(group)).to_numpy()
    counts = sp.csr_matrix(adata.layers[counts_layer])[mask]
    library = np.asarray(counts.sum(axis=1)).ravel().astype(float)
    dispersion, trend = _nb_dispersion(counts, library)
    out = pd.DataFrame({
        'mean_count': np.asarray(counts.sum(axis=0)).ravel() / mask.sum(),
        'dispersion': dispersion,
        'dispersion_trend': trend,
    }, index=adata.var_names)
    if symbol_col in adata.var.columns:
        out.insert(0, 'symbol', adata.var[symbol_col].astype(str).values)
    return out


def dropout_diagnostic(
    adata,                        # AnnData: cells or metacells with a raw counts layer and model gene var_names.
    leverage,                     # DataFrame: output of gene_dropout_leverage.
    groupby,                      # str: adata.obs column holding cell types.
    counts_layer='counts',        # str: layer with raw integer counts.
    detection_prob=0.95,          # float: target probability of detecting a gene.
    reference_size=50,            # int: metacell size to benchmark against (e.g. SEACells target_metacell_size).
    model='nb'                    # str: 'nb' (negative binomial, expression-matched dispersion) or 'poisson'.
):
    """
    Per gene and cell type, compares observed zeros to the zeros expected from sequencing depth
    and typical cell-to-cell variability, and estimates how many median-depth units must be pooled
    to detect the gene with probability `detection_prob`.

    With model='poisson', every cell of a type is assumed to share one expression rate, so any
    variability (bursting, technical noise, subpopulations) shows up as excess zeros; this
    over-flags well-expressed genes. With model='nb', expected zeros allow the amount of extra
    variability that is typical for genes at that expression level in that cell type, so only
    genes that are unusually uneven are flagged.

    Observed zeros close to expected means dropout is sampling-driven and pooling will fix it.
    Observed zeros well above expected means the gene is unusually uneven between cells of that
    type (possibly a subpopulation), so which cells get pooled together matters.

    Returns (per-gene DataFrame, per-cell-type summary DataFrame), with summaries weighted by leverage.
    """
    if model not in ('nb', 'poisson'):
        raise ValueError(f"model must be 'nb' or 'poisson', got '{model}'")
    counts = sp.csr_matrix(adata.layers[counts_layer])
    library = np.asarray(counts.sum(axis=1)).ravel().astype(float)
    genes = [g for g in leverage.index if g in adata.var_names]
    gene_idx = adata.var_names.get_indexer(genes)
    labels = adata.obs[groupby].astype(str).values

    rows, summary = [], []
    for group in pd.unique(labels):
        if group not in leverage.columns:
            continue
        mask = labels == group
        group_counts = counts[mask]
        sub = group_counts[:, gene_idx]
        lib = library[mask]
        n_units = int(mask.sum())

        if model == 'nb':
            dispersion = _nb_dispersion(group_counts, lib)[1][gene_idx]
        else:
            dispersion = np.zeros(len(genes))

        zero_frac = 1 - sub.getnnz(axis=0) / n_units
        rate = np.asarray(sub.sum(axis=0)).ravel() / lib.sum()
        expected_zero = sum(_p_zero(np.outer(lib[i:i + 2000], rate), dispersion).sum(axis=0)
                            for i in range(0, n_units, 2000)) / n_units

        # Pooling s independent median-depth cells: P(all zero) = p_zero(median-depth mean) ** s
        per_cell_zero = _p_zero(rate * np.median(lib), dispersion)
        with np.errstate(divide='ignore', invalid='ignore'):
            units_needed = np.where(rate > 0, np.log(1 - detection_prob) / np.log(per_cell_zero), np.inf)

        weight = leverage.loc[genes, group].to_numpy(dtype=float)
        rows.append(pd.DataFrame({
            'group': group, 'model_gene': genes, 'symbol': leverage.loc[genes, 'symbol'].values,
            'leverage': weight, 'zero_frac': zero_frac, 'expected_zero_frac': expected_zero,
            'excess_zero_frac': zero_frac - expected_zero, 'units_needed': units_needed,
            'rate': rate, 'dispersion': dispersion,
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


# =========================================================
# 5. CELL QC AND UNEXPECTED ZEROS
# =========================================================
def cell_qc(adata, counts_layer='counts', symbol_col='gene_symbol', mito_prefix='mt-'):
    """Per-cell library size, genes detected and percentage of counts from mitochondrial genes."""
    counts = sp.csr_matrix(adata.layers[counts_layer])
    symbols = adata.var[symbol_col].astype(str) if symbol_col in adata.var.columns else adata.var_names.to_series().astype(str)
    mito = symbols.str.lower().str.startswith(mito_prefix.lower()).to_numpy()
    total = np.asarray(counts.sum(axis=1)).ravel()
    mito_total = np.asarray(counts[:, mito].sum(axis=1)).ravel()
    return pd.DataFrame({
        'total_counts': total,
        'n_genes': np.diff(counts.indptr),
        'pct_mito': 100 * mito_total / np.maximum(total, 1),
    }, index=adata.obs_names)


def unexpected_zeros(
    adata,                        # AnnData: cells with a raw counts layer and model gene var_names.
    genes,                        # list: model gene IDs to check.
    groupby,                      # str: adata.obs column holding cell types.
    group,                        # str: the cell type to check.
    counts_layer='counts',        # str: layer with raw integer counts.
    min_detect_prob=0.9,          # float: a zero only counts as unexpected if detection was predicted at least this likely.
    model='nb'                    # str: 'nb' or 'poisson', as in dropout_diagnostic.
):
    """
    For each cell in `group` and each gene, flags zeros that should not have happened given the
    cell's depth, the cell type's average expression rate and (for model='nb') the variability
    typical at that expression level: detection probability >= min_detect_prob. If the same cells carry unexpected zeros across many genes, the zeros are a
    property of those cells (a subpopulation, or damaged/low-quality cells) rather than random dropout.

    Returns (unexpected: cells x genes bool DataFrame, detect_prob: cells x genes float DataFrame).
    """
    if model not in ('nb', 'poisson'):
        raise ValueError(f"model must be 'nb' or 'poisson', got '{model}'")
    mask = (adata.obs[groupby].astype(str) == str(group)).to_numpy()
    group_counts = sp.csr_matrix(adata.layers[counts_layer])[mask]
    library = np.asarray(group_counts.sum(axis=1)).ravel().astype(float)
    genes = [g for g in genes if g in adata.var_names]
    gene_idx = adata.var_names.get_indexer(genes)
    sub = group_counts[:, gene_idx].toarray()

    dispersion = _nb_dispersion(group_counts, library)[1][gene_idx] if model == 'nb' else np.zeros(len(genes))
    rate = sub.sum(axis=0) / library.sum()
    detect_prob = 1 - _p_zero(np.outer(library, rate), dispersion)
    unexpected = (sub == 0) & (detect_prob >= min_detect_prob)

    cells = adata.obs_names[mask]
    return (pd.DataFrame(unexpected, index=cells, columns=genes),
            pd.DataFrame(detect_prob, index=cells, columns=genes))


# =========================================================
# 6. METACELL SIZE TARGETS
# =========================================================
def metacell_size_targets(
    gene_df,                      # DataFrame: per-gene output of dropout_diagnostic.
    summary,                      # DataFrame: per-cell-type output of dropout_diagnostic.
    coverage=0.8,                 # float: share of reachable leverage each metacell should detect.
    min_metacells=3               # int: fewest metacells to keep per cell type (sets the size cap).
):
    """
    Turns the dropout diagnostic into a metacell size rule per cell type.

    The cap is n_cells / min_metacells, so every cell type keeps at least `min_metacells` metacells
    (and therefore some within-type variation). Genes needing more cells than the cap are
    'unreachable': pooling cannot fix them without merging most of the cell type, so they are
    listed for downstream handling (e.g. wider flux bounds) instead of driving metacell size.

    The target is the smallest size at which `coverage` of the reachable leverage is detected with
    the dropout_diagnostic detection probability. It is also given as a UMI budget
    (target cells x median library size), which is what an adaptive method should aim for,
    since deeper cells need fewer partners.

    Returns (targets DataFrame per cell type, uncertain genes DataFrame).
    """
    rows, uncertain = [], []
    for group, s in summary.iterrows():
        n_cells = int(s['n_units'])
        median_lib = float(s['median_library_size'])
        cap = max(1, n_cells // min_metacells)

        d = gene_df[(gene_df['group'] == group) & (gene_df['leverage'] > 0)]
        needed = d['units_needed'].to_numpy(dtype=float)
        weight = d['leverage'].to_numpy(dtype=float)
        total = weight.sum()
        reachable = needed <= cap

        if reachable.any():
            target = _weighted_quantile(needed[reachable], weight[reachable], coverage)
            target = int(np.clip(np.ceil(target), 1, cap))
        else:
            target = cap

        rows.append({
            'group': group,
            'n_cells': n_cells,
            'median_library_size': median_lib,
            'cap_cells': cap,
            'target_cells': target,
            'target_umis': int(round(target * median_lib)),
            'n_metacells': n_cells // target,
            'leverage_covered_at_target': float(weight[needed <= target].sum() / total) if total else np.nan,
            'leverage_unreachable': float(weight[~reachable].sum() / total) if total else np.nan,
            'n_unreachable_genes': int((~reachable).sum()),
        })
        unreachable = d[~reachable].assign(cap_cells=cap)
        uncertain.append(unreachable)

    uncertain = pd.concat(uncertain, ignore_index=True) if uncertain else pd.DataFrame()
    return pd.DataFrame(rows).set_index('group'), uncertain
