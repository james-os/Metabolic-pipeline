import re
import importlib.resources as pkg_resources


def resolve_model_path(model_path):
    """Return a real file path, swapping "default" for the packaged mitoMAMMAL model."""
    if model_path == "default":
        return str(pkg_resources.files('metabolic_tools') / 'data' / 'mitoMAMMALmod.json')
    return model_path


def model_symbol_to_id(model_json, species_prefix=None):
    """
    Builds a gene symbol -> model gene ID map from the GENE_ASSOCIATION / GENE_LIST
    reaction notes. Symbols that point to more than one ID are dropped as ambiguous.
    """
    pairs = {}
    for rxn in model_json.get('reactions', []):
        notes = {str(k).lower(): str(v) for k, v in rxn.get('notes', {}).items()}
        assoc_str = notes.get('gene_association', '')
        list_str = notes.get('gene_list', '')
        if not assoc_str or not list_str:
            continue

        c_assoc = re.sub(r'(?i)\b(and|or)\b', ' ', re.sub(r'[\(\)\[\],]', ' ', assoc_str)).split()
        c_list = re.sub(r'(?i)\b(and|or)\b', ' ', re.sub(r'[\(\)\[\],]', ' ', list_str)).split()
        if len(c_assoc) != len(c_list):
            continue

        for gid, sym in zip(c_assoc, c_list):
            if species_prefix and not gid.startswith(species_prefix):
                continue
            pairs.setdefault(sym, set()).add(gid)

    return {sym: next(iter(ids)) for sym, ids in pairs.items() if len(ids) == 1}


def map_var_names_to_model(adata, model_json, symbol_col, species_prefix=None):
    """
    If none of adata.var_names are model gene IDs (e.g. the index holds symbols or
    placeholder IDs), rename var_names to model IDs by matching adata.var[symbol_col]
    against the model's gene symbols. Unmatched genes keep their original name.
    Returns the (possibly renamed) AnnData.
    """
    model_ids = set()
    for rxn in model_json.get('reactions', []):
        rule = str(rxn.get('gene_reaction_rule', ''))
        model_ids.update(t for t in re.findall(r'[A-Za-z0-9\-\.]+', rule) if t.lower() not in ('and', 'or'))
    if species_prefix:
        model_ids = {g for g in model_ids if g.startswith(species_prefix)}

    if any(g in model_ids for g in adata.var_names):
        return adata

    if symbol_col and symbol_col in adata.var.columns:
        symbols = adata.var[symbol_col].astype(str)
    else:
        symbols = adata.var_names.astype(str).to_series(index=adata.var_names)

    sym_to_id = model_symbol_to_id(model_json, species_prefix)
    new_names, used = [], set(adata.var_names)
    n_mapped = 0
    for orig, sym in zip(adata.var_names, symbols):
        gid = sym_to_id.get(sym)
        if gid and gid not in used:
            new_names.append(gid)
            used.add(gid)
            n_mapped += 1
        else:
            new_names.append(orig)

    if n_mapped == 0:
        return adata

    source = f"adata.var['{symbol_col}']" if symbol_col in adata.var.columns else "var_names"
    print(f"--> var_names contain no model gene IDs; mapped {n_mapped} genes to model IDs via {source}.")
    adata = adata.copy()
    if symbol_col and symbol_col not in adata.var.columns:
        adata.var[symbol_col] = adata.var_names.astype(str)
    adata.var_names = new_names
    return adata
