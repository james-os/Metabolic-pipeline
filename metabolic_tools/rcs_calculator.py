import scanpy as sc
import cobra
import json
import numpy as np
import pandas as pd
import anndata 
from anndata import AnnData
import re
import os
import scipy.sparse as sp

def calculate_rcs_matrix(adata_path, model_path, qc_json_path, 
                         and_strategy='mean', or_strategy='sum',
                         output_path=None):
    """
    Calculates Reaction Capacity Scores (RCS) with dynamic species awareness,
    single-gene hub filtering, AND Global GPR Redundancy filtering to prevent
    matrix inflation from duplicated model rules.
    """
    
    print("=== Initiating RCS Calculation ===")
    
    # 1. Load Files
    try:
        adata = sc.read_h5ad(adata_path)
        model = cobra.io.load_json_model(model_path)
        with open(qc_json_path, 'r') as f:
            qc_report = json.load(f)
    except Exception as e:
        raise IOError(f"Failed to load files: {e}")

    original_species_gene_pool = set(adata.var_names)

    # 2. Apply "Treatment" using QC JSON
    print("Applying metabolic cleaning filters from JSON...")
    genes_to_drop = []
    hub_genes_retained = set() 
    
    for ensembl_id, metadata in qc_report.get("gene_details", {}).items():
        if metadata.get("QC_Status") == "Zero Expression":
            genes_to_drop.append(ensembl_id)
        elif metadata.get("is_hub_artifact") == True:
            hub_genes_retained.add(ensembl_id)
            
    valid_genes = [g for g in adata.var_names if g not in genes_to_drop]
    adata = adata[:, valid_genes].copy()
    
    print(f"Dropped {len(genes_to_drop)} zero-expression genes.")
    print(f"Tracked {len(hub_genes_retained)} flagged 'hub' genes.")

    # 3. Define Math Operations
    math_ops = {
        'min': lambda x: np.min(x, axis=0),
        'max': lambda x: np.max(x, axis=0),
        'mean': lambda x: np.mean(x, axis=0),
        'sum': lambda x: np.sum(x, axis=0)
    }
    
    func_and = math_ops[and_strategy]
    func_or = math_ops[or_strategy]

    expr_matrix = adata.X.toarray() if sp.issparse(adata.X) else adata.X
    gene_expr_dict = {gene: expr_matrix[:, i] for i, gene in enumerate(adata.var_names)}

    # 4. Helper Function: Evaluate GPR Logic
    def evaluate_gpr_array(rule_string):
        if not rule_string:
            return np.zeros(adata.n_obs)
            
        rule_string = rule_string.replace('(', '').replace(')', '')
        or_blocks = re.split(r'\s+or\s+', rule_string, flags=re.IGNORECASE)
        or_arrays = []
        
        for block in or_blocks:
            and_genes = re.split(r'\s+and\s+', block, flags=re.IGNORECASE)
            and_arrays = []
            for gene in and_genes:
                gene = gene.strip()
                if gene in gene_expr_dict:
                    and_arrays.append(gene_expr_dict[gene])
                else:
                    and_arrays.append(np.zeros(adata.n_obs))
            
            block_result = func_and(and_arrays) if and_arrays else np.zeros(adata.n_obs)
            or_arrays.append(block_result)
            
        final_result = func_or(or_arrays) if or_arrays else np.zeros(adata.n_obs)
        return final_result

    # 5. Calculate Reaction Capacity Scores (RCS)
    print(f"Calculating RCS using AND={and_strategy}, OR={or_strategy}...")
    reaction_ids = []
    reaction_matrix = []
    
    used_single_gene_hubs = set()
    used_gpr_signatures = set() # NEW: Registry for identical multi-gene rules
    
    skipped_hub_reactions = 0
    skipped_redundant_rules = 0

    for rxn in model.reactions:
        rule_string = rxn.gene_reaction_rule
        if not rule_string:
            continue
            
        # --- NEW: Global Redundancy Filter ---
        # Standardize the string (lowercase, remove extra spaces/parentheses)
        standardized_rule = re.sub(r'\s+', ' ', rule_string.lower().replace('(', '').replace(')', '').strip())
        
        if standardized_rule in used_gpr_signatures:
            # We already calculated this exact GPR rule for a different reaction.
            skipped_redundant_rules += 1
            continue
        
        # --- Dynamic Species Hub Filter ---
        tokens = [t.strip() for t in re.split(r'\s+and\s+|\s+or\s+|\(|\)', rule_string, flags=re.IGNORECASE) if t.strip()]
        valid_species_genes = list(set([g for g in tokens if g in original_species_gene_pool]))
        
        is_effective_single_gene = (len(valid_species_genes) == 1)
        
        if is_effective_single_gene:
            effective_gene = valid_species_genes[0]
            if effective_gene in hub_genes_retained:
                if effective_gene in used_single_gene_hubs:
                    skipped_hub_reactions += 1
                    continue
                else:
                    used_single_gene_hubs.add(effective_gene)

        # Drop down to mathematical calculation
        rcs_array = evaluate_gpr_array(rxn.gene_reaction_rule)
        
        if np.any(rcs_array > 0):
            reaction_ids.append(rxn.id)
            reaction_matrix.append(rcs_array)
            # Register this rule so it can never spawn a duplicate column again
            used_gpr_signatures.add(standardized_rule)

    # 6. Build New AnnData Object
    rcs_matrix = np.vstack(reaction_matrix).T
    
    rcs_adata = AnnData(X=rcs_matrix)
    rcs_adata.obs = adata.obs.copy()  
    rcs_adata.var_names = reaction_ids
    
    print(f"=== RCS Matrix Complete ===")
    print(f"Intercepted {skipped_hub_reactions} redundant single-gene hub reactions.")
    print(f"Intercepted {skipped_redundant_rules} duplicated GPR rules (matrix inflation prevented).")
    print(f"Final shape: {rcs_adata.n_obs} observations x {rcs_adata.n_vars} unique capabilities")

    # 7. Save Output 
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        anndata.settings.allow_write_nullable_strings = False 
        rcs_adata.write_h5ad(output_path)
        print(f"Saved RCS matrix to {output_path}")

    return rcs_adata