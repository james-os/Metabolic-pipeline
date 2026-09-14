import scanpy as sc
import cobra
import numpy as np
import scipy.sparse as sp
import pandas as pd
import os
import json
import re
from datetime import datetime
from .gene_mapping import resolve_model_path, map_var_names_to_model


def cleaning_report(
    adata_path: str, 
    model_path: str, 
    species: str = 'mmusculus', 
    gene_column: str = 'gene_symbol', 
    output_dir: str = None
) -> dict:
    """
    Cleans transcriptomic data against the mitoMAMMAL JSON model natively keyed by Ensembl IDs.
    Utilizes a specified gene_column from the dataset to cleanly annotate the final reports.
    Categorizes species-specific GPR logic (Single, Complex, Isozyme, Mixed) 
    and flags genes duplicating across independent pathways.
    """
    prefix_map = {'hsapiens': 'ENSG', 'mmusculus': 'ENSMUSG', 'drerio': 'ENSDARG'}
    if species not in prefix_map:
        raise ValueError(f"Invalid species argument '{species}'")
    species_prefix = prefix_map[species]
    
    print(f"\n=== Initiating Metabolic Data Cleaning Pipeline ({species} / mitoMAMMAL) ===")
    
    # =========================================================================
    # STEP 1: PARSE RAW JSON TO BUILD GENE MAP AND CATEGORIZE GPR RULES
    # =========================================================================
    if model_path == "default":
        print("--> Loading default modified mitoMAMMAL model from package resources...")
    else:
        print(f"--> Loading custom model from {model_path}...")
    # Looks inside the 'metabolic_tools/data' folder for the JSON file when "default"
    model_path = resolve_model_path(model_path)
    with open(model_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    id_to_symbol = {}
    rxn_to_category = {}

    for rxn in raw_data.get('reactions', []):
        rxn_id = rxn.get('id')
        notes = rxn.get('notes', {})
        notes_lower = {str(k).lower(): str(v) for k, v in notes.items()}
        
        assoc_str = notes_lower.get('gene_association', '')
        list_str = notes_lower.get('gene_list', '')

        if not assoc_str:
            rxn_to_category[rxn_id] = "none"
            continue

        # 1A. Build the Fallback Symbol Dictionary
        c_assoc = re.sub(r'[\(\)\[\],]', ' ', assoc_str)
        c_list = re.sub(r'[\(\)\[\],]', ' ', list_str)
        c_assoc = re.sub(r'(?i)\b(and|or)\b', ' ', c_assoc)
        c_list = re.sub(r'(?i)\b(and|or)\b', ' ', c_list)

        raw_ids = c_assoc.split()
        raw_syms = c_list.split()

        if len(raw_ids) == len(raw_syms):
            for gid, sym in zip(raw_ids, raw_syms):
                if gid.startswith(species_prefix):
                    id_to_symbol[gid.strip()] = sym.strip()

        # 1B. Isolate the Species-Specific Rule Block
        blocks = re.split(r'\)\s*or\s*\(', assoc_str, flags=re.IGNORECASE)
        species_rule = ""
        for block in blocks:
            if species_prefix in block:
                species_rule = block
                break
        
        if not species_rule:
            species_rule = assoc_str

        # 1C. Categorize the GPR logic strictly within the target species
        species_genes = [w for w in re.sub(r'[\(\)\[\],]', ' ', species_rule).split() if w.startswith(species_prefix)]
        gene_count = len(set(species_genes))

        if gene_count == 0:
            rxn_to_category[rxn_id] = "none"
        elif gene_count == 1:
            rxn_to_category[rxn_id] = "single_gene"
        else:
            has_and = " and " in species_rule.lower()
            has_or = " or " in species_rule.lower()
            
            if has_and and has_or:
                rxn_to_category[rxn_id] = "mixed_and_or"
            elif has_and:
                rxn_to_category[rxn_id] = "complex_and"
            elif has_or:
                rxn_to_category[rxn_id] = "isozyme_or"
            else:
                rxn_to_category[rxn_id] = "single_gene"

    print(f"--> Extracted {len(id_to_symbol)} fallback Ensembl-to-Symbol pairs from JSON.")

    # =========================================================================
    # STEP 2: STANDARD COBRAPY & SCANPY INTERSECTION
    # =========================================================================
    print("Loading transcriptomics and building model objects...")
    try:
        adata = sc.read_h5ad(adata_path)
        model = cobra.io.load_json_model(model_path)
    except Exception as e:
        raise IOError(f"Failed to load files: {e}")

    # Datasets indexed by symbols or placeholder IDs are mapped onto the model's Ensembl IDs
    adata = map_var_names_to_model(adata, raw_data, gene_column, species_prefix)

    # --- NEW: Extract dataset-specific gene annotations ---
    adata_symbol_map = {}
    if gene_column and gene_column != 'index':
        if gene_column in adata.var.columns:
            print(f"--> Using '{gene_column}' from adata.var to annotate gene symbols.")
            adata_symbol_map = dict(zip(adata.var.index, adata.var[gene_column].astype(str)))
        else:
            print(f"--> WARNING: Column '{gene_column}' not found in adata.var. Falling back to JSON symbols.")

    is_sparse = sp.issparse(adata.X)
    max_val = float(adata.X.max())
    is_logged = max_val < 50
    print(f"Normalization State: {'Appears Log-Normalized' if is_logged else 'Appears as Raw Counts'} (Max expression = {max_val:.2f})")

    target_genes = [g for g in model.genes if g.id.startswith(species_prefix)]
    model_gene_ids = [g.id for g in target_genes]
    adata_genes = set(adata.var_names)
    
    mapped_ids = list(set(model_gene_ids).intersection(adata_genes))
    missing_ids = list(set(model_gene_ids).difference(adata_genes))

    if not mapped_ids:
        raise ValueError(
            f"None of the {len(model_gene_ids)} {species} model genes were found in the dataset. "
            f"Check that species='{species}' is correct and that adata.var_names (or "
            f"adata.var['{gene_column}']) holds Ensembl IDs or gene symbols."
        )

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
    print("Compiling artifact risks based on species-isolated GPRs...")
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
            
        # --- NEW: Prefer mapping from the dataset; fallback to the parsed JSON; fallback to Ensembl ID ---
        true_name = adata_symbol_map.get(clean_id, id_to_symbol.get(clean_id, clean_id))
        
        all_rxns = []
        single_gene_rxns = []
        complex_rxns = []
        isozyme_rxns = []
        mixed_rxns = []

        for r in g.reactions:
            cat = rxn_to_category.get(r.id, "none")
            if cat == "none": continue 
            
            all_rxns.append(r.id)
            if cat == "single_gene": single_gene_rxns.append(r.id)
            elif cat == "complex_and": complex_rxns.append(r.id)
            elif cat == "isozyme_or": isozyme_rxns.append(r.id)
            elif cat == "mixed_and_or": mixed_rxns.append(r.id)
        
        independent_driver_count = len(single_gene_rxns) + len(isozyme_rxns)
        is_hub_artifact = independent_driver_count > 1
        
        if is_hub_artifact:
            hub_artifact_count += 1
        
        gene_metadata_list.append({
            "Ensembl_ID": clean_id,
            "Gene_Symbol": true_name,
            "QC_Status": status,
            "Total_Reactions": len(all_rxns),
            "Single_Gene_GPRs": len(single_gene_rxns),
            "Isozyme_OR_GPRs": len(isozyme_rxns),
            "Complex_AND_GPRs": len(complex_rxns),
            "Mixed_GPRs": len(mixed_rxns),
            "Independent_Driver_Count": independent_driver_count,
            "Hub_Artifact_Risk": is_hub_artifact,
            "Affected_Reactions": " | ".join(all_rxns)
        })

        gene_metadata_dict[clean_id] = {
            "ensembl_id": clean_id,
            "gene_symbol": true_name,
            "status": status,
            "total_reactions": len(all_rxns),
            "rule_categories": {
                "single_gene": single_gene_rxns,
                "isozyme_or": isozyme_rxns,
                "complex_and": complex_rxns,
                "mixed_and_or": mixed_rxns
            },
            "is_hub_artifact": is_hub_artifact
        }

    print(f"--> Detected {hub_artifact_count} Hub genes duplicating across single/isozyme GPRs.")

    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": "mitoMAMMAL (JSON)",
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
        
        json_path = os.path.join(output_dir, "metabolic_cleaning_automated.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
            
        df = pd.DataFrame(gene_metadata_list)
        df.sort_values(by=["Hub_Artifact_Risk", "Independent_Driver_Count"], ascending=[False, False], inplace=True)
        df.to_csv(os.path.join(output_dir, "metabolic_cleaning_summary.csv"), index=False)

    print("=== Cleaning Complete ===")
    return results
