import scanpy as sc
import cobra
import numpy as np
import scipy.sparse as sp
import pandas as pd
import os
import json
import re
from datetime import datetime

def metabolic_qc_report(adata_path: str, model_path: str, species: str = 'mmusculus', output_dir: str = None) -> dict:
    """
    Performs QC and automatically extracts hidden gene symbols from a raw JSON model.
    Precisely flags genes exclusively duplicated across multiple single-gene GPRs 
    to prevent downstream clustering artifacts.
    """
    prefix_map = {'hsapiens': 'ENSG', 'mmusculus': 'ENSMUSG', 'drerio': 'ENSDARG'}
    if species not in prefix_map:
        raise ValueError(f"Invalid species argument '{species}'")
    species_prefix = prefix_map[species]
    
    print("\n=== Initiating Metabolic QC Pipeline ===")
    
    # =========================================================================
    # STEP 1: MANUALLY BUILD THE GENE MAP FROM RAW JSON
    # =========================================================================
    print(f"Scanning raw JSON to build Gene Symbol map...")
    with open(model_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    id_to_symbol = {}
    
    for rxn in raw_data.get('reactions', []):
        notes = rxn.get('notes', {})
        notes_lower = {str(k).lower(): str(v) for k, v in notes.items()}
        
        if 'gene_association' in notes_lower and 'gene_list' in notes_lower:
            assoc_str = notes_lower['gene_association']
            list_str = notes_lower['gene_list']

            c_assoc = re.sub(r'[\(\)\[\],]', ' ', assoc_str)
            c_list = re.sub(r'[\(\)\[\],]', ' ', list_str)
            c_assoc = re.sub(r'(?i)\b(and|or)\b', ' ', c_assoc)
            c_list = re.sub(r'(?i)\b(and|or)\b', ' ', c_list)

            raw_ids = c_assoc.split()
            raw_syms = c_list.split()

            if len(raw_ids) == len(raw_syms):
                for gid, sym in zip(raw_ids, raw_syms):
                    id_to_symbol[gid.strip()] = sym.strip()

    print(f"--> Successfully extracted {len(id_to_symbol)} unique Ensembl-to-Symbol pairs.")

    # =========================================================================
    # STEP 2: STANDARD COBRAPY & SCANPY QC
    # =========================================================================
    print("Loading transcriptomics and building model objects...")
    try:
        adata = sc.read_h5ad(adata_path)
        model = cobra.io.load_json_model(model_path)
    except Exception as e:
        raise IOError(f"Failed to load files: {e}")

    is_sparse = sp.issparse(adata.X)
    max_val = float(adata.X.max())
    is_logged = max_val < 50
    print(f"Normalization State: {'Appears Log-Normalized' if is_logged else 'Appears as Raw Counts'} (Max expression = {max_val:.2f})")

    target_genes = [g for g in model.genes if g.id.startswith(species_prefix)]
    model_gene_ids = [g.id for g in target_genes]
    adata_genes = set(adata.var_names)
    
    mapped_ids = list(set(model_gene_ids).intersection(adata_genes))
    missing_ids = list(set(model_gene_ids).difference(adata_genes))
        
    metabolic_adata = adata[:, mapped_ids]
    if is_sparse:
        nnz = metabolic_adata.X.nnz
        total_elements = metabolic_adata.shape[0] * metabolic_adata.shape[1]
        gene_sums = np.array(metabolic_adata.X.sum(axis=0)).flatten()
    else:
        nnz = np.count_nonzero(metabolic_adata.X)
        total_elements = metabolic_adata.X.size
        gene_sums = np.sum(metabolic_adata.X, axis=0)
        
    global_sparsity = 1.0 - (nnz / total_elements)
    zero_expression_mask = gene_sums == 0
    zero_expr_ids = np.array(mapped_ids)[zero_expression_mask].tolist()

    # =========================================================================
    # STEP 3: BUILD THE DUAL OUTPUT REPORTS (CSV & JSON)
    # =========================================================================
    print("Compiling artifact risks and final reports...")
    gene_metadata_list = []
    gene_metadata_dict = {}
    hub_artifact_count = 0
    
    for g in target_genes:
        clean_id = g.id.strip()
        
        if clean_id in zero_expr_ids:
            status = "Zero Expression"
        elif clean_id in missing_ids:
            status = "Missing from Dataset"
        else:
            status = "Mapped & Expressed"
            
        true_name = id_to_symbol.get(clean_id, clean_id)
        
        # --- PRECISION ARTIFACT LOGIC ---
        reactions_list = [r.name for r in g.reactions]
        
        # If the reaction's rule does NOT contain "and", it acts as a single-gene driver (pure OR)
        single_gene_rxns = [r.name for r in g.reactions if " and " not in r.gene_reaction_rule.lower()]
        
        # If the reaction's rule contains "and", it is part of a complex, which mitigates duplication risk
        complex_rxns = [r.name for r in g.reactions if " and " in r.gene_reaction_rule.lower()]
        
        # A gene is only a toxic hub artifact if it independently drives MORE than 1 reaction
        is_hub_artifact = len(single_gene_rxns) > 1
        
        if is_hub_artifact:
            hub_artifact_count += 1
        
        gene_metadata_list.append({
            "Ensembl_ID": clean_id,
            "Gene_Name": true_name,
            "QC_Status": status,
            "Total_Reactions": len(reactions_list),
            "Single_Gene_GPRs": len(single_gene_rxns),
            "Complex_AND_GPRs": len(complex_rxns),
            "Hub_Artifact_Risk": is_hub_artifact,
            "Affected_Reactions": " | ".join(reactions_list)
        })

        gene_metadata_dict[clean_id] = {
            "ensembl_id": clean_id,
            "gene_name": true_name,
            "status": status,
            "total_reactions": len(reactions_list),
            "single_gene_gprs": len(single_gene_rxns),
            "complex_and_gprs": len(complex_rxns),
            "is_hub_artifact": is_hub_artifact,
            "reactions": reactions_list
        }

    print(f"--> Detected {hub_artifact_count} Hub genes exclusively duplicating across single-gene GPRs.")

    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "species_analyzed": species,
        "is_logged": is_logged,
        "max_expression": max_val,
        "is_sparse": is_sparse,
        "total_model_genes": len(model_gene_ids),
        "mapped_genes_count": len(mapped_ids),
        "missing_genes_count": len(missing_ids),
        "metabolic_dropout_rate": global_sparsity,
        "zero_expression_genes_count": len(zero_expr_ids),
        "hub_artifact_genes_count": hub_artifact_count,
        "gene_details": gene_metadata_dict
    }

    if output_dir:
        print(f"\nSaving reports to: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        json_path = os.path.join(output_dir, "metabolic_qc_automated.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
            
        df = pd.DataFrame(gene_metadata_list)
        # Sort so the most severe pure-duplication artifacts sit at the top
        df.sort_values(by=["Hub_Artifact_Risk", "Single_Gene_GPRs"], ascending=[False, False], inplace=True)
        df.to_csv(os.path.join(output_dir, "metabolic_qc_summary.csv"), index=False)

    print("=== QC Complete ===")
    return results