import os
import re
import ast
import json
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import anndata
import importlib.resources as pkg_resources

def calculate_ecs(
    adata_path, 
    model_path, 
    cleaning_report_path, 
    output_dir,
    model_gene_col=None,       
    symbol_col='gene_symbol',  
    split_isozymes=True,
    and_strategy='min', 
    or_strategy='sum'
):
    """
    Calculates the ECS matrix by parsing the mitoMAMMAL JSON model.
    Implements a Global Redundancy Registry to ensure identical GPR rules 
    are only calculated once, protecting hub genes from deletion.
    """
    print("--> Loading transcriptomic data...")
    adata = sc.read_h5ad(adata_path)
    
    if model_path == "default":
        print("--> Loading default modified mitoMAMMAL model from package resources...")
        with pkg_resources.path('metabolic_tools', 'mitoMAMMAL_modified.json') as default_path:
            with open(default_path, 'r', encoding='utf-8') as f:
                model_json = json.load(f)
    else:
        print(f"--> Loading custom model natively from {model_path}...")
        with open(model_path, 'r', encoding='utf-8') as f:
            model_json = json.load(f)

    # Safely map Ensembl IDs to Gene Symbols
    ensembl_ids = adata.var.index if model_gene_col is None else adata.var[model_gene_col]
    symbols = adata.var[symbol_col] if symbol_col in adata.var.columns else adata.var.index
    
    ensembl_to_symbol = {str(k): str(v) for k, v in zip(ensembl_ids, symbols)}
    symbol_to_ensembl = {str(v): str(k) for k, v in zip(ensembl_ids, symbols)}

    print("--> Processing metabolic cleaning report...")
    with open(cleaning_report_path, 'r') as f:
        report_data = json.load(f)
        
    cleaning_df = pd.DataFrame.from_dict(report_data['gene_details'], orient='index')
    if 'gene_symbol' not in cleaning_df.columns:
        cleaning_df['gene_symbol'] = cleaning_df.index
    
    # --- THE FIX: We NO LONGER drop hub artifacts here! Only true missing genes. ---
    bad_genes_mask = (cleaning_df['status'] == 'Missing from Dataset')
    bad_symbols = set(cleaning_df.loc[bad_genes_mask, 'gene_symbol'])
    bad_model_genes = {symbol_to_ensembl[sym] for sym in bad_symbols if sym in symbol_to_ensembl}
    
    reactions_to_split = set()
    if split_isozymes:
        for rules in cleaning_df['rule_categories'].dropna():
            if isinstance(rules, dict) and 'isozyme_or' in rules:
                reactions_to_split.update(rules['isozyme_or'])

    print("--> Extracting expression arrays...")
    expr_dict = {}
    var_names = ensembl_ids.astype(str).tolist()
    
    matrix = adata.X.toarray() if sp.issparse(adata.X) else adata.X
    n_cells = matrix.shape[0]
    
    for i, gene in enumerate(var_names):
        if gene in expr_dict:
            expr_dict[gene] = np.maximum(expr_dict[gene], matrix[:, i])
        else:
            expr_dict[gene] = matrix[:, i]
        
    def evaluate_gpr_node(rule_string):
        if not rule_string or pd.isna(rule_string):
            return np.zeros(n_cells)
            
        tokens = re.findall(r'[a-zA-Z0-9\-\.]+', str(rule_string))
        unique_genes = set(t for t in tokens if t.lower() not in ['and', 'or'])
        
        safe_rule = str(rule_string)
        gene_mapping = {}
        for i, gene in enumerate(unique_genes):
            safe_name = f"VAR_{i}"
            gene_mapping[safe_name] = gene
            pattern = rf'(?<![a-zA-Z0-9\-\.]){re.escape(gene)}(?![a-zA-Z0-9\-\.])'
            safe_rule = re.sub(pattern, safe_name, safe_rule)
            
        safe_rule = re.sub(r'\bAND\b', 'and', safe_rule, flags=re.IGNORECASE)
        safe_rule = re.sub(r'\bOR\b', 'or', safe_rule, flags=re.IGNORECASE)
        
        try:
            tree = ast.parse(safe_rule, mode='eval').body
        except SyntaxError:
            return np.zeros(n_cells)
            
        def _eval(node):
            if isinstance(node, ast.Name):
                original_gene = gene_mapping.get(node.id, node.id)
                return expr_dict.get(original_gene, np.zeros(n_cells))
            elif isinstance(node, ast.BoolOp):
                arrays = [_eval(val) for val in node.values]
                if isinstance(node.op, ast.And):
                    return np.min(arrays, axis=0) if and_strategy == 'min' else np.mean(arrays, axis=0)
                elif isinstance(node.op, ast.Or):
                    return np.sum(arrays, axis=0) if or_strategy == 'sum' else np.max(arrays, axis=0)
            return np.zeros(n_cells)
            
        return _eval(tree)

    ecs_features = {}
    feature_metadata = {} 
    
    # --- NEW: Registry to track calculated rules and prevent duplicates ---
    seen_gpr_signatures = set()
    
    print("--> Calculating Enzymatic Capacity Scores...")
    for rxn in model_json.get('reactions', []):
        if not isinstance(rxn, dict):
            continue
            
        notes = rxn.get('notes', {})
        if isinstance(notes, dict) and 'GENE_ASSOCIATION' in notes:
            rule = str(notes.get('GENE_ASSOCIATION', ''))
        else:
            rule = str(rxn.get('gene_reaction_rule', ''))
            
        if not rule or rule == 'nan':
            continue
            
        rxn_id = str(rxn.get('id', 'Unknown_Reaction'))
        
        subsystem = rxn.get('subsystem', 'Unknown')
        if isinstance(subsystem, list):
            subsystem = subsystem[0] if subsystem else 'Unknown'
        subsystem = str(subsystem)
            
        tokens = re.findall(r'[a-zA-Z0-9\-\.]+', rule)
        genes_in_rule = set(t for t in tokens if t.lower() not in ['and', 'or'])
        
        mouse_genes_in_rule = {g for g in genes_in_rule if g in ensembl_to_symbol}
        
        # If all mouse genes are missing from dataset, skip
        if mouse_genes_in_rule and mouse_genes_in_rule.issubset(bad_model_genes):
            continue 
            
        # --- NEW: Redundancy Signature Check ---
        # Create a mathematical fingerprint for this rule (Sorted genes + logic operators)
        has_and = ' and ' in rule.lower()
        has_or = ' or ' in rule.lower()
        rule_signature = (tuple(sorted(mouse_genes_in_rule)), has_and, has_or)
        
        # If we have already calculated this exact combination, skip it to prevent matrix bloat
        if rule_signature in seen_gpr_signatures:
            continue
            
        seen_gpr_signatures.add(rule_signature)
            
        def generate_label(branch_string):
            if ' and ' in branch_string.lower():
                return rxn_id
                
            b_tokens = re.findall(r'[a-zA-Z0-9\-\.]+', branch_string)
            b_genes = [t for t in b_tokens if t.lower() not in ['and', 'or']]
            
            mouse_symbols = []
            for g in b_genes:
                if g in ensembl_to_symbol:
                    sym = ensembl_to_symbol[g]
                    if sym not in mouse_symbols:
                        mouse_symbols.append(sym)
            
            if len(mouse_symbols) == 1:
                return str(mouse_symbols[0])
            elif len(mouse_symbols) > 1:
                s1 = min(mouse_symbols)
                s2 = max(mouse_symbols)
                prefix = ""
                for i, c in enumerate(s1):
                    if i < len(s2) and c == s2[i]:
                        prefix += c
                    else:
                        break
                
                if len(prefix) >= 2:
                    return f"{prefix}*"
                
            return rxn_id
            
        if split_isozymes and rxn_id in reactions_to_split and ' or ' in rule.lower():
            branches = re.split(r'\s+or\s+', rule, flags=re.IGNORECASE)
            
            for i, branch in enumerate(branches):
                clean_branch = branch.strip("() ")
                
                branch_genes = re.findall(r'[a-zA-Z0-9\-\.]+', clean_branch)
                if not any(g in ensembl_to_symbol for g in branch_genes):
                    continue 
                
                score_array = evaluate_gpr_node(clean_branch)
                feature_label = generate_label(clean_branch)
                
                if feature_label == rxn_id and len(branches) > 1:
                    feature_label = f"{rxn_id}_iso{i+1}"
                
                final_name = f"{rxn_id} ({feature_label} isozyme)" if feature_label != rxn_id else feature_label
                
                ecs_features[final_name] = score_array
                feature_metadata[final_name] = {'subsystem': subsystem, 'original_reaction_id': rxn_id}
        else:
            score_array = evaluate_gpr_node(rule)
            feature_label = generate_label(rule)
            
            ecs_features[feature_label] = score_array
            feature_metadata[feature_label] = {'subsystem': subsystem, 'original_reaction_id': rxn_id}
            
    # --- NEW: Post-Calculation Vector Deduplication ---
    print("--> Scrubbing identical feature vectors...")
    unique_ecs_features = {}
    seen_arrays = []
    
    for label, score_array in ecs_features.items():
        # Check if this exact array has been generated by a previous reaction/split
        is_duplicate = any(np.array_equal(score_array, seen) for seen in seen_arrays)
        
        if not is_duplicate:
            unique_ecs_features[label] = score_array
            seen_arrays.append(score_array)
        else:
            # Clean up the metadata registry to prevent misalignment
            if label in feature_metadata:
                del feature_metadata[label]
                
    print("--> Compiling final ECS matrix...")
    ecs_df = pd.DataFrame(unique_ecs_features, index=adata.obs_names)
    
    var_df = pd.DataFrame.from_dict(feature_metadata, orient='index')
    
    ecs_adata = sc.AnnData(X=ecs_df.values, obs=adata.obs.copy())
    ecs_adata.var_names = ecs_df.columns
    ecs_adata.var = var_df.loc[ecs_adata.var_names].copy()
    ecs_adata.var_names_make_unique()
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "ecs_matrix_isozyme_split.h5ad")
    
    anndata.settings.allow_write_nullable_strings = False
    ecs_adata.write(out_path)