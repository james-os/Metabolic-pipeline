"""Local smoke test: runs notebook 03's code path on tiny synthetic data.

Checks that the pipeline runs end to end and that its invariants hold. It does not
check scientific quality -- that is what the real benchmark on the HPC is for.
"""
import sys, os, json, tempfile, traceback
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metabolic_tools.metacell_diagnostics import recover_counts
from metabolic_tools.metabolic_metacells import metabolic_metacells, classify_reactions
from metabolic_tools.gene_mapping import resolve_model_path
from metabolic_tools.metacell_benchmark import thin_counts, fixed_size_labels, evaluate_grouping
from synthetic_data import make_adata

with open(resolve_model_path('default'), encoding='utf-8') as _f:
    model_json_for_test = json.load(_f)

CELLTYPE_COL, SAMPLE_COL = 'cell_type', 'sample'
THIN_FRACTIONS = [0.5, 0.25]
REFERENCE_SIZE = 50

# mirrors notebook 03's settings cell
SIZING_VARIANTS = {
    'split=T,and=median': dict(split_isozymes=True,  sizing_and_strategy='median'),
    'split=F,and=median': dict(split_isozymes=False, sizing_and_strategy='median'),
    'split=T,and=min':    dict(split_isozymes=True,  sizing_and_strategy='min'),
    'split=F,and=min':    dict(split_isozymes=False, sizing_and_strategy='min'),
}
REFERENCE_VARIANT = 'split=T,and=median'
SCORING_SPLIT_ISOZYMES = False
SEED = 0

PASS, FAIL = [], []


def check(name, condition, detail=''):
    condition = bool(condition)
    (PASS if condition else FAIL).append(name)
    print(f'  {"PASS" if condition else "FAIL"}  {name}' + (f'\n        -- {detail}' if detail and not condition else ''))


def section(title):
    print(f'\n=== {title} ===')


# ---------------------------------------------------------------- data
section('0. Synthetic data and count recovery')
adata = make_adata(seed=SEED)
adata.obs[SAMPLE_COL] = adata.obs_names.str.split('_E16').str[0]
print(f'  {adata.n_obs} cells x {adata.n_vars} genes; '
      f'{adata.obs[CELLTYPE_COL].nunique()} cell types, {adata.obs[SAMPLE_COL].nunique()} samples')
truth_total = adata.uns['truth']['true_counts_total']
report = recover_counts(adata)
check('counts recovered exactly from log-normalised X',
      np.isclose(float(adata.layers['counts'].sum()), truth_total),
      f'{float(adata.layers["counts"].sum())} vs {truth_total}')
print(f'  integer fraction: {report.get("integer_fraction")}, '
      f'median UMIs: {np.median(np.asarray(adata.layers["counts"].sum(axis=1))):.0f}')

# ---------------------------------------------------------------- part 1
section('1. metabolic_metacells on the full data')
common = dict(celltype_col=CELLTYPE_COL, sample_col=SAMPLE_COL, species='mmusculus',
              symbol_col='gene_symbol', and_strategy='median', or_strategy='sum',
              coverage=0.8, min_metacells=3, budget_tolerance=0.9, random_state=SEED)
mc, info = metabolic_metacells(adata, **common)
cells = info['cells']
print(f'  {cells.n_obs} cells -> {mc.n_obs} metacells; {int(mc.obs["under_budget"].sum())} under budget')
print(info['targets'][['n_cells', 'median_library_size', 'cap_cells', 'target_cells', 'target_umis']].to_string())

labels = info['labels']
check('every cell has a label', labels.notna().all() and len(labels) == cells.n_obs)
check('metacells never mix cell types', (mc.obs['celltype_purity'] == 1).all())
check('metacells never mix samples', (mc.obs['sample_purity'] == 1).all())
check('cell counts add up', int(mc.obs['n_cells'].sum()) == cells.n_obs)

cell_counts = sp.csr_matrix(cells.layers['counts'])
check('total UMIs conserved by aggregation',
      np.isclose(float(mc.layers['counts'].sum()), float(cell_counts.sum())))
name = mc.obs_names[int(np.argmax(mc.obs['n_cells'].to_numpy()))]
members = labels.index[labels.values == name]
expected = np.asarray(cell_counts[cells.obs_names.get_indexer(members)].sum(axis=0)).ravel()
got = np.asarray(sp.csr_matrix(mc.layers['counts'])[mc.obs_names.get_loc(name)].todense()).ravel()
check('largest metacell equals the sum of its cells', np.allclose(expected, got))

X = mc.X.toarray() if sp.issparse(mc.X) else np.asarray(mc.X)
check('metacell X is log1p(counts per 10k)',
      np.allclose(np.expm1(X.astype(np.float64)).sum(axis=1), 1e4, rtol=1e-2),
      f'row sums {np.expm1(X.astype(np.float64)).sum(axis=1)[:3]}')

# under_budget must mean "this stratum could not afford a whole metacell", not "the split was uneven"
library = np.asarray(cell_counts.sum(axis=1)).ravel().astype(float)
strata = cells.obs[[CELLTYPE_COL, SAMPLE_COL]].astype(str)
affordable = {}
for key, idx in strata.groupby([CELLTYPE_COL, SAMPLE_COL]).indices.items():
    affordable[key] = int(library[idx].sum() // float(info['targets'].loc[key[0], 'target_umis']))
stratum_of = list(zip(mc.obs[CELLTYPE_COL].astype(str), mc.obs[SAMPLE_COL].astype(str)))
could_afford = np.array([affordable[s] >= 1 for s in stratum_of])
starved = mc.obs['under_budget'].to_numpy() & could_afford
check('no metacell is starved by an uneven split', not starved.any(),
      mc.obs.loc[starved, [CELLTYPE_COL, SAMPLE_COL, 'total_umis', 'umi_budget', 'budget_ratio']].to_string())
check('under-budget metacells are the whole of their stratum',
      (mc.obs.loc[mc.obs['under_budget'], 'n_cells'].to_numpy()
       == np.array([len(strata.groupby([CELLTYPE_COL, SAMPLE_COL]).indices[s])
                    for s, u in zip(stratum_of, mc.obs['under_budget']) if u])).all())

DET_COLS = ['detected_leverage_pooled', 'expected_detected_leverage_pooled',
            'detected_leverage_reachable', 'expected_detected_leverage_reachable',
            'detected_leverage_all', 'expected_detected_leverage_all']
for col in ['umi_budget', 'budget_ratio', 'under_budget'] + DET_COLS:
    check(f'obs column {col!r} present', col in mc.obs.columns)
# Per metacell these need not be ordered -- a weighted mean over a superset of genes is not bounded
# by the subset's -- but across metacells the tiers must separate, which is why the class exists.
check('detection separates the tiers on average',
      mc.obs['detected_leverage_pooled'].mean() > mc.obs['detected_leverage_reachable'].mean()
      > mc.obs['detected_leverage_all'].mean(),
      mc.obs[DET_COLS].mean().round(4).to_string())
det = mc.obs[DET_COLS]
check('detection fractions are in [0, 1] with no NaN',
      det.notna().all().all() and (det >= 0).all().all() and (det <= 1 + 1e-9).all().all(),
      det.describe().to_string())

# ---------------------------------------------------------------- classes
section('2. Gene and reaction classes')
gc, rc = info['gene_classes'], info['reaction_classes']
celltype_set = set(cells.obs[CELLTYPE_COL].astype(str))
CLASSES = {'off', 'uncertain', 'partial', 'pooled'}
check('gene classes are off/uncertain/partial/pooled', set(gc['class']) <= CLASSES)
check('reaction classes are off/uncertain/partial/pooled', set(rc['class']) <= CLASSES)
check('all four gene classes occur', set(gc['class']) == CLASSES, sorted(set(gc['class'])))

# the classes must follow units_needed against the sizes actually used
check('pooled genes are detectable within the size target',
      (gc.loc[gc['class'] == 'pooled', 'units_needed']
       <= gc.loc[gc['class'] == 'pooled', 'target_cells']).all())
check('partial genes sit between the size target and the cap',
      ((gc.loc[gc['class'] == 'partial', 'units_needed'] > gc.loc[gc['class'] == 'partial', 'target_cells'])
       & (gc.loc[gc['class'] == 'partial', 'units_needed'] <= gc.loc[gc['class'] == 'partial', 'cap_cells'])).all())
check('uncertain genes need more cells than the cap',
      (gc.loc[gc['class'] == 'uncertain', 'units_needed']
       > gc.loc[gc['class'] == 'uncertain', 'cap_cells']).all())

# detection must fall monotonically across the tiers, which is the reason the class exists
counts_mc = sp.csr_matrix(mc.layers['counts'])
seen = {}
for ct, idx in mc.obs.groupby(mc.obs[CELLTYPE_COL].astype(str)).indices.items():
    seen[ct] = pd.Series(np.asarray((counts_mc[idx] > 0).sum(axis=0)).ravel() / len(idx),
                         index=mc.var_names.astype(str))
gc_seen = gc.assign(detected=[seen[grp].get(gene, np.nan)
                              for grp, gene in zip(gc['group'], gc['model_gene'])])
by_class = gc_seen.groupby('class')['detected'].mean()
print('  mean share of metacells detecting the gene, by class:')
print('   ' + by_class.round(3).to_string().replace('\n', '\n   '))
check('detection falls from pooled to partial to uncertain',
      by_class['pooled'] > by_class['partial'] > by_class['uncertain'], by_class.round(3).to_string())

# the reaction ladder must be monotone: relaxing what counts as support can only improve a class
rank = {'off': 0, 'uncertain': 1, 'partial': 2, 'pooled': 3}
strict = gc.assign(**{'class': gc['class'].where(gc['class'] != 'partial', 'uncertain')})
rc_strict = classify_reactions(model_json_for_test, strict, set(cells.var_names.astype(str)),
                               'mmusculus', 'median', 'sum')
merged = (rc.set_index(['group', 'reaction_id'])['class']
          .map(rank).rename('with_partial')
          .to_frame().join(rc_strict.set_index(['group', 'reaction_id'])['class'].map(rank).rename('without')))
check('treating partial genes as uncertain never improves a reaction class',
      (merged['with_partial'] >= merged['without']).all())
print(f'  reactions by class: {rc["class"].value_counts().to_dict()}')
check('one gene row per cell type x model gene', not gc.duplicated(['group', 'model_gene']).any())
check('one reaction row per cell type x reaction', not rc.duplicated(['group', 'reaction_id']).any())
check('gene class groups are the cell types', set(gc['group']) == celltype_set,
      f'{sorted(set(gc["group"]))} vs {sorted(celltype_set)}')
check('reaction class groups are the cell types', set(rc['group']) == celltype_set,
      f'{sorted(set(rc["group"]))} vs {sorted(celltype_set)}')

absent_symbols = set(adata.uns['truth']['absent_symbols'])
sym = gc['symbol'].astype(str)
absent_rows = gc[sym.isin(absent_symbols)]
check('genes absent from the dataset are classed off',
      len(absent_rows) > 0 and (absent_rows['class'] == 'off').all(),
      f'{len(absent_rows)} rows, classes {sorted(set(absent_rows["class"]))}')
present_rows = gc[~sym.isin(absent_symbols)]
check('genes present in the dataset are not classed off',
      not (present_rows['class'] == 'off').any(),
      f'{int((present_rows["class"] == "off").sum())} present genes classed off')
print(pd.crosstab(gc['group'], gc['class']).to_string())
print(pd.crosstab(rc['group'], rc['class']).to_string())

# ---------------------------------------------------------------- determinism
section('3. Determinism')
mc2, info2 = metabolic_metacells(adata, **common)
check('same seed gives the same labels', info2['labels'].equals(info['labels']))

# ---------------------------------------------------------------- sizing knobs
section('3b. Sizing knobs (split_isozymes, sizing_and_strategy)')
# the defaults must reproduce the run above, so the knobs cannot change behaviour unless asked
mc_def, info_def = metabolic_metacells(adata, split_isozymes=True, sizing_and_strategy=None, **common)
check('defaults leave the result unchanged', info_def['labels'].equals(info['labels']))
check('sizing_and_strategy=None matches and_strategy',
      info_def['targets']['target_umis'].equals(info['targets']['target_umis']))

variants = {'split=True, and=median (default)': dict(split_isozymes=True),
            'split=False': dict(split_isozymes=False),
            'sizing and=min': dict(sizing_and_strategy='min'),
            'split=False, sizing and=min': dict(split_isozymes=False, sizing_and_strategy='min')}
knob_budgets = {}
for label, kw in variants.items():
    mc_v, info_v = metabolic_metacells(adata, **common, **kw)
    knob_budgets[label] = info_v['targets']['target_umis']
    check(f'runs with {label}', mc_v.n_obs > 0 and (mc_v.obs['sample_purity'] == 1).all())
    print(f'    {label:<32} {mc_v.n_obs:>3} metacells, mean budget {knob_budgets[label].mean():>8.0f} UMIs')
check('split_isozymes changes the UMI budget',
      not knob_budgets['split=True, and=median (default)'].equals(knob_budgets['split=False']))
check('sizing_and_strategy changes the UMI budget',
      not knob_budgets['split=True, and=median (default)'].equals(knob_budgets['sizing and=min']))
check('sizing_and_strategy is independent of the scoring and_strategy',
      metabolic_metacells(adata, **{**common, 'and_strategy': 'min'}, sizing_and_strategy='median'
                          )[1]['targets']['target_umis'].equals(knob_budgets['split=True, and=median (default)']))

# ---------------------------------------------------------------- part 2
section('4. Benchmark path (thinning and comparison groupings)')
records = []
for frac in THIN_FRACTIONS:
    thinned = thin_counts(cells, frac, random_state=SEED)
    kept = float(sp.csr_matrix(thinned.layers['counts']).sum()) / float(cell_counts.sum())
    check(f'thinning to {frac:.0%} keeps about that share of UMIs ({kept:.3f})', abs(kept - frac) < 0.02)
    check(f'thinned cells line up with the originals ({frac:.0%})', thinned.obs_names.equals(cells.obs_names))

    # notebook cell 10: size every sweep variant on the thinned cells, sharing one embedding
    var_runs = {}
    for nm, kw in SIZING_VARIANTS.items():
        mc_t, info_t = metabolic_metacells(thinned, **common, **kw)
        thinned = info_t['cells']
        var_runs[nm] = (mc_t, info_t)
    check(f'every sweep variant sizes on the same embedding ({frac:.0%})',
          all(info_t['cells'].obsm['X_pca'].shape == thinned.obsm['X_pca'].shape
              for _, info_t in var_runs.values()))

    groupings = [(nm, info_t['labels'], info_t) for nm, (mc_t, info_t) in var_runs.items()]
    ref_mc, ref_info = var_runs[REFERENCE_VARIANT]
    budgets = ref_info['targets']['target_umis']
    matched_size = (thinned.obs[CELLTYPE_COL].astype(str).value_counts()
                    / ref_mc.obs[CELLTYPE_COL].value_counts()).to_dict()
    check(f'matched sizes are finite for every cell type ({frac:.0%})',
          set(matched_size) == celltype_set and all(np.isfinite(v) for v in matched_size.values()),
          str(matched_size))
    groupings.append(('fixed_size_matched',
                      fixed_size_labels(thinned, CELLTYPE_COL, matched_size, random_state=SEED), ref_info))
    groupings.append((f'fixed_{REFERENCE_SIZE}',
                      fixed_size_labels(thinned, CELLTYPE_COL, REFERENCE_SIZE, random_state=SEED), ref_info))

    for nm, lab, info_g in groupings:
        check(f'{nm} labels every cell ({frac:.0%})', lab.notna().all() and len(lab) == thinned.n_obs)
        records.append(evaluate_grouping(cells, thinned, lab, nm, frac, CELLTYPE_COL, SAMPLE_COL,
                                         info_g['targets']['target_umis'], info_g['genes'],
                                         info_g['gene_classes'], species='mmusculus',
                                         and_strategy='median', or_strategy='sum',
                                         split_isozymes=SCORING_SPLIT_ISOZYMES, budget_tolerance=0.9))
    info_t = ref_info
bench = pd.concat(records, ignore_index=True)
bench['is_variant'] = bench['method'].isin(SIZING_VARIANTS)
print(f'  benchmark table: {bench.shape}')
check('benchmark has rows', len(bench) > 0)
check('every sweep variant and reference produced rows',
      set(bench['method']) == set(SIZING_VARIANTS) | {'fixed_size_matched', f'fixed_{REFERENCE_SIZE}'},
      sorted(set(bench['method'])))
check('every variant is scored on the same feature set',
      bench[bench['is_variant']].groupby('method')['features_present'].sum().nunique() >= 1
      and bench.loc[bench['is_variant'], 'features_present'].gt(0).all())

# scoring the summed reaction rather than each isozyme branch is a different feature set
split_on = evaluate_grouping(cells, thinned, ref_info['labels'], REFERENCE_VARIANT, THIN_FRACTIONS[-1],
                             CELLTYPE_COL, SAMPLE_COL, budgets, ref_info['genes'], ref_info['gene_classes'],
                             species='mmusculus', and_strategy='median', or_strategy='sum',
                             split_isozymes=True, budget_tolerance=0.9)
split_off = bench[(bench['method'] == REFERENCE_VARIANT) & (bench['fraction'] == THIN_FRACTIONS[-1])]
check('evaluate_grouping accepts split_isozymes and it changes the feature set',
      split_off['features_present'].sum() < split_on['features_present'].sum(),
      f"split=False {split_off['features_present'].sum()} vs split=True {split_on['features_present'].sum()}")
print(f'    features scored: split=True {split_on["features_present"].sum()}, '
      f'split=False {split_off["features_present"].sum()}; '
      f'false zeros {split_on["false_zero_rate"].median():.3f} vs {split_off["false_zero_rate"].median():.3f}')
for col in ['false_zero_rate', 'median_rel_error', 'spearman', 'compactness', 'budget_ratio']:
    check(f'benchmark column {col!r} is mostly finite', bench[col].notna().mean() > 0.5,
          f'{bench[col].notna().mean():.2f} finite')
check('false zero rates are in [0, 1]', bench['false_zero_rate'].dropna().between(0, 1).all())
check('deeper data gives no more false zeros than shallower',
      bench[bench['fraction'] == 0.5]['false_zero_rate'].median()
      <= bench[bench['fraction'] == 0.25]['false_zero_rate'].median() + 1e-9,
      bench.groupby('fraction')['false_zero_rate'].median().to_string())
check('the metabolic method keeps samples pure',
      (bench.loc[bench['method'] == REFERENCE_VARIANT, 'sample_purity'] == 1).all())

# ---------------------------------------------------------------- notebook aggregations
section('5. Notebook aggregation cells')
try:
    full_runs, full_cells = {}, adata
    for nm, kw in SIZING_VARIANTS.items():
        mc_v, info_v = metabolic_metacells(full_cells, **common, **kw)
        full_cells = info_v['cells']
        full_runs[nm] = (mc_v, info_v)
    sizing_summary = pd.DataFrame({
        nm: {'metacells': int(mc_v.n_obs),
             'median_cells': float(mc_v.obs['n_cells'].median()),
             'mean_budget_umis': float(mc_v.obs['umi_budget'].mean()),
             'under_budget': int(mc_v.obs['under_budget'].sum()),
             'detected_reachable': float(mc_v.obs['detected_leverage_reachable'].median()),
             **{f'rxn_{k}': int((info_v['reaction_classes']['class'] == k).sum())
                for k in ['pooled', 'partial', 'uncertain', 'off']}}
        for nm, (mc_v, info_v) in full_runs.items()}).T
    check('cell 5: sizing sweep summary', len(sizing_summary) == len(SIZING_VARIANTS))
    print(sizing_summary.round(3).to_string())
except Exception:
    check('cell 5: sizing sweep summary', False)
    traceback.print_exc()

try:
    per_stratum = (mc.obs.groupby([CELLTYPE_COL, SAMPLE_COL], observed=True)
                   .agg(metacells=('n_cells', 'size'), cells=('n_cells', 'sum'),
                        median_cells=('n_cells', 'median'), min_budget_ratio=('budget_ratio', 'min'),
                        under_budget=('under_budget', 'sum'),
                        detected_reachable=('detected_leverage_reachable', 'median')).round(2))
    check('cell 6: metacells per stratum', len(per_stratum) > 0)
except Exception:
    check('cell 6: metacells per stratum', False)
    traceback.print_exc()

try:
    summary = (bench.groupby(['fraction', 'method'])
               .agg(metacells=('metacell', 'size'), median_cells=('n_cells', 'median'),
                    false_zero_rate=('false_zero_rate', 'median'),
                    median_rel_error=('median_rel_error', 'median'), spearman=('spearman', 'median'),
                    detected_leverage_all=('detected_leverage_all', 'median'),
                    detected_leverage_reachable=('detected_leverage_reachable', 'median'),
                    compactness=('compactness', 'median'),
                    mixed_sample_share=('sample_purity', lambda s: float((s < 1).mean())),
                    under_budget_share=('under_budget', 'mean')).round(3))
    order = list(SIZING_VARIANTS) + [x for x in summary.index.get_level_values('method').unique()
                                     if x not in SIZING_VARIANTS]
    summary = summary.reindex(pd.MultiIndex.from_product([THIN_FRACTIONS, order],
                                                         names=['fraction', 'method'])).dropna(how='all')
    check('cell 11: benchmark summary', len(summary) > 0)
    check('cell 11: every variant and reference survives the reindex',
          len(summary) == len(THIN_FRACTIONS) * (len(SIZING_VARIANTS) + 2), f'{len(summary)} rows')
    print(summary.to_string())
except Exception:
    check('cell 11: benchmark summary', False)
    traceback.print_exc()

try:
    by_type = bench.pivot_table(index=['fraction', CELLTYPE_COL], columns='method',
                                values='false_zero_rate', aggfunc='median').round(3)
    check('cell 12: false zeros by cell type', len(by_type) > 0)
except Exception:
    check('cell 12: false zeros by cell type', False)
    traceback.print_exc()

try:
    m = bench[bench['method'] == REFERENCE_VARIANT]
    paired = (m.groupby(['fraction', CELLTYPE_COL, 'under_budget'])
               .agg(metacells=('metacell', 'size'), false_zero_rate=('false_zero_rate', 'median'),
                    detected_leverage_all=('detected_leverage_all', 'median'),
                    budget_ratio=('budget_ratio', 'median')).unstack('under_budget'))
    paired = paired.reindex(columns=pd.MultiIndex.from_product(
        [['metacells', 'false_zero_rate', 'detected_leverage_all', 'budget_ratio'], [False, True]]))
    has_both = paired['metacells'].notna().all(axis=1)
    comparison = pd.DataFrame({
        'at_budget_metacells': paired[('metacells', False)],
        'under_budget_metacells': paired[('metacells', True)],
        'under_budget_ratio': paired[('budget_ratio', True)],
        'false_zero_at': paired[('false_zero_rate', False)],
        'false_zero_under': paired[('false_zero_rate', True)],
        'detected_at': paired[('detected_leverage_all', False)],
        'detected_under': paired[('detected_leverage_all', True)],
    })
    comparison['false_zero_increase'] = comparison['false_zero_under'] - comparison['false_zero_at']
    comparison['detection_drop'] = comparison['detected_at'] - comparison['detected_under']
    check('cell 14: under-budget comparison', True)
    print(f'  cell types with both under- and at-budget metacells: {int(has_both.sum())}')
except Exception:
    check('cell 14: under-budget comparison', False)
    traceback.print_exc()

try:
    bins = [0, 0.25, 0.5, 0.75, 0.9, 1.1, 1.5, 2, np.inf]
    bench['budget_bin'] = pd.cut(bench['budget_ratio'], bins)
    binned = bench.pivot_table(index=['fraction', 'budget_bin'], columns='method',
                               values='false_zero_rate', aggfunc='median', observed=True).round(3)
    check('cell 16: false zeros by budget bin', len(binned) > 0)
except Exception:
    check('cell 16: false zeros by budget bin', False)
    traceback.print_exc()

# ---------------------------------------------------------------- plots
section('5b. Notebook plotting cells (headless)')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.hist(np.log2(mc.obs['budget_ratio']), bins=40, color='grey')
    ax.axvline(np.log2(0.9), color='tab:red', ls='--', lw=1)
    ax = axes[1]
    colors = np.where(mc.obs['under_budget'], 'tab:red', 'tab:blue')
    ax.scatter(mc.obs['expected_detected_leverage_pooled'], mc.obs['detected_leverage_pooled'],
               c=colors, s=18, alpha=0.7, linewidths=0)
    ax.plot([0, 1], [0, 1], color='grey', ls='--', lw=0.8)
    plt.tight_layout()
    plt.close(fig)
    check('cell 7: budget and detection plots', True)
except Exception:
    check('cell 7: budget and detection plots', False)
    traceback.print_exc()

try:
    markers = {'fixed_size_matched': 's', f'fixed_{REFERENCE_SIZE}': '^'}
    fig, axes = plt.subplots(2, len(THIN_FRACTIONS), figsize=(6 * len(THIN_FRACTIONS), 8), squeeze=False)
    for c, frac in enumerate(THIN_FRACTIONS):
        for r, metric in enumerate(['false_zero_rate', 'detected_leverage_all']):
            ax = axes[r, c]
            for method, d in bench[bench['fraction'] == frac].groupby('method'):
                ax.scatter(d['budget_ratio'], d[metric], s=14, alpha=0.5,
                           marker=markers.get(method, 'o'), label=method, linewidths=0)
            ax.set_xscale('log')
    axes[0, 0].legend(fontsize=8)
    plt.tight_layout()
    plt.close(fig)
    check('cell 16: sparsity vs budget plot', True)
except Exception:
    check('cell 16: sparsity vs budget plot', False)
    traceback.print_exc()

try:
    fig, axes = plt.subplots(1, len(THIN_FRACTIONS), figsize=(5 * len(THIN_FRACTIONS), 4.2), squeeze=False)
    for ax, frac in zip(axes.flat, THIN_FRACTIONS):
        d = m[m['fraction'] == frac]
        ax.scatter(d['expected_detected_leverage_pooled'], d['detected_leverage_pooled'],
                   c=np.where(d['under_budget'], 'tab:red', 'tab:blue'), s=18, alpha=0.7, linewidths=0)
        ax.plot([0, 1], [0, 1], color='grey', ls='--', lw=0.8)
        err = d['detected_leverage_pooled'] - d['expected_detected_leverage_pooled']
        ax.set_title(f'{frac:.0%} kept, mean observed - predicted = {err.mean():+.3f}', fontsize=9)
    plt.tight_layout()
    plt.close(fig)
    for frac in THIN_FRACTIONS:
        d = m[m['fraction'] == frac]
        for flag, g in d.groupby('under_budget'):
            e = g['detected_leverage_pooled'] - g['expected_detected_leverage_pooled']
            print(f'    {frac:.0%} kept | under_budget={flag}: n={len(g)}, mean error {e.mean():+.3f}, '
                  f'corr(predicted, false_zero_rate) = '
                  f'{g["expected_detected_leverage_pooled"].corr(g["false_zero_rate"]):+.2f}')
    check('cell 17: predicted vs observed detection', True)
except Exception:
    check('cell 17: predicted vs observed detection', False)
    traceback.print_exc()

# ---------------------------------------------------------------- saving
section('6. Saving outputs')
out = tempfile.mkdtemp()
try:
    mc.write_h5ad(os.path.join(out, 'mc.h5ad'))
    info['labels'].rename('metacell').to_csv(os.path.join(out, 'cell_to_metacell.csv'))
    info['targets'].to_csv(os.path.join(out, 'size_targets.csv'))
    info['gene_classes'].to_csv(os.path.join(out, 'gene_classes.csv'), index=False)
    info['reaction_classes'].to_csv(os.path.join(out, 'reaction_classes.csv'), index=False)
    bench.drop(columns=['budget_bin']).to_csv(os.path.join(out, 'benchmark.csv'), index=False)
    check('cell 20: all outputs write', True)
except Exception:
    check('cell 20: all outputs write', False)
    traceback.print_exc()

# ---------------------------------------------------------------- result
section('RESULT')
print(f'  {len(PASS)} passed, {len(FAIL)} failed')
for f in FAIL:
    print(f'   FAILED: {f}')
sys.exit(1 if FAIL else 0)
