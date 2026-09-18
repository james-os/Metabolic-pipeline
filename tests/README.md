# Smoke test

`smoke_test.py` runs the whole code path of
`notebooks/03_metabolic_metacells_benchmark.ipynb` on a tiny synthetic dataset, so a broken
environment or a regression shows up in about a minute rather than part-way through a real run.

```bash
python tests/smoke_test.py
```

It exits non-zero if any check fails, so it also works as a post-install check on a new
environment (for example after building the venv on CREATE, before running notebook 03).

`synthetic_data.py` builds the dataset: ~600 cells over 5 cell types and 3 samples of differing
depth, using the real mouse gene symbols from the packaged mitoMAMMAL model so the GPR rules
actually fire, with some model genes absent from every cell so the `off` class is exercised. `X`
is log1p counts-per-10k with no counts layer, so `recover_counts` has to reconstruct the counts
as it does on the real Kolla data.

What is checked: exact count recovery, UMI conservation through aggregation, metacells never
mixing cell types or samples, `under_budget` meaning only that a stratum could not afford a whole
metacell, gene and reaction classes, determinism under a fixed seed, binomial thinning, all three
comparison groupings, every aggregation and plotting cell in notebook 03, and saving the outputs.

It checks that the pipeline behaves as designed, not that the method is scientifically right --
the real benchmark on Kolla E16 is what answers that. Nothing here needs SEACells; the
`seacells_labels` grouping is not covered.
