import importlib

# Public functions are imported on first use, so optional heavy dependencies
# (e.g. SEACells) are only needed by the tools that actually use them. Without this
# the package cannot be imported at all on a machine that has no SEACells, which
# stops the model-audit tools running even though they never touch it.
_EXPORTS = {
    'stratify_metacells': '.seacell_aggregation',
    'cleaning_report': '.cleaning_report',
    'calculate_ecs': '.ecs_calculator',
    'resolution_sweep': '.cluster_checker',
    'celltype_annotate': '.module_1',
    'characterise_metabolism': '.module_2',
    'audit_model_genes': '.model_gene_audit',
    'build_decisions': '.model_gene_patch',
    'apply_decisions': '.model_gene_patch',
    'compare_variants': '.model_variant_benchmark',
    'resolve': '.paths',
    'register_dataset': '.paths',
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
