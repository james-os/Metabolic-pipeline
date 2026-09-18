"""Tiny synthetic dataset shaped like Kolla E16: model gene symbols, several cell
types and samples, uneven depth, and some genes absent from the whole dataset."""
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
from metabolic_tools.gene_mapping import resolve_model_path, model_symbol_to_id

CELLTYPES = {'IHC': 250, 'OHC_1': 150, 'GER': 120, 'LER_Fst': 55, 'Hensen': 22}
SAMPLES = {'se_1': 0.45, 'se_2': 1.0, 'se_3': 1.4}   # relative sequencing depth
N_PADDING = 200
N_ABSENT = 30                                        # model genes with no counts anywhere


def make_adata(seed=0, median_umis=2700):
    rng = np.random.default_rng(seed)
    model_json = json.load(open(resolve_model_path('default'), encoding='utf-8'))
    symbols = sorted(model_symbol_to_id(model_json, 'ENSMUSG'))
    padding = [f'Pad{i:04d}' for i in range(N_PADDING)]
    all_symbols = symbols + padding
    n_genes = len(all_symbols)

    # cell type x sample composition, deliberately uneven (as in the real data)
    comp = rng.dirichlet(np.full(len(SAMPLES), 2.0), size=len(CELLTYPES))
    obs_rows, type_of_cell, sample_of_cell = [], [], []
    for t, (celltype, n) in enumerate(CELLTYPES.items()):
        counts = rng.multinomial(n, comp[t])
        for s, sample in enumerate(SAMPLES):
            for k in range(counts[s]):
                obs_rows.append(f'{sample}_E16_{celltype}_{k}')
                type_of_cell.append(celltype)
                sample_of_cell.append(sample)
    n_cells = len(obs_rows)
    type_of_cell = np.array(type_of_cell)
    sample_of_cell = np.array(sample_of_cell)

    # gene rates: a shared baseline plus per-cell-type modulation, so types differ
    base = rng.lognormal(mean=-4.0, sigma=1.6, size=n_genes)
    base /= base.sum()
    rates = np.tile(base, (len(CELLTYPES), 1))
    for t in range(len(CELLTYPES)):
        shift = rng.lognormal(mean=0.0, sigma=0.8, size=n_genes)
        rates[t] *= shift
        rates[t] /= rates[t].sum()

    # some model genes are absent everywhere (liver-type enzymes in the real data)
    absent = rng.choice(len(symbols), size=N_ABSENT, replace=False)
    rates[:, absent] = 0.0
    rates /= rates.sum(axis=1, keepdims=True)

    depth = np.array([SAMPLES[s] for s in sample_of_cell])
    library = rng.lognormal(mean=np.log(median_umis), sigma=0.45, size=n_cells) * depth
    library = np.maximum(np.round(library), 200).astype(int)

    type_index = {c: i for i, c in enumerate(CELLTYPES)}
    rows = []
    for i in range(n_cells):
        p = rates[type_index[type_of_cell[i]]]
        draw = rng.poisson(library[i] * p * rng.gamma(shape=4.0, scale=0.25, size=n_genes))
        rows.append(sp.csr_matrix(draw.astype(np.float64)))
    counts = sp.vstack(rows).tocsr()
    counts.eliminate_zeros()

    # log1p(counts per 10k), and no counts layer -- recover_counts must reconstruct it
    total = np.asarray(counts.sum(axis=1)).ravel()
    norm = sp.csr_matrix(sp.diags(1e4 / np.maximum(total, 1)) @ counts)
    norm.data = np.log1p(norm.data)

    var = pd.DataFrame({'gene_symbol': all_symbols}, index=[f'GENE{i:05d}' for i in range(n_genes)])
    obs = pd.DataFrame({'cell_type': pd.Categorical(type_of_cell)}, index=obs_rows)
    adata = ad.AnnData(X=norm.astype(np.float32), obs=obs, var=var)
    adata.uns['truth'] = {'absent_symbols': [symbols[i] for i in absent],
                          'true_counts_total': float(counts.sum())}
    return adata
