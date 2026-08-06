import scanpy as sc
import cobra
import json
import numpy as np
import pandas as pd
import anndata  # Added this import to access settings
from anndata import AnnData
import re
import os
import scipy.sparse as sp

def calculate_rcs_matrix(adata_path, model_path, qc_json_path,
                         and_strategy='mean', or_strategy='sum',
                         output_path=None):
    """
    Cleans scRNA-seq (or metacell) data using a QC JSON report and calculates
    Reaction Capacity Scores (RCS) based on flexible GPR math.
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

    # 2. Apply "Treatment" using QC JSON
    print("Applying metabolic cleaning filters from JSON...")
    genes_to_drop = []

    for ensembl_id, metadata in qc_report.get("gene_details", {}).items():
        # Drop genes with universal 0 expression or flagged as hub artifacts
        if metadata.get("is_hub_artifact") == True or metadata.get("QC_Status") == "Zero Expression":
            genes_to_drop.append(ensembl_id)

    # Filter the AnnData object to only include safe genes
    valid_genes = [g for g in adata.var_names if g not in genes_to_drop]
    adata = adata[:, valid_genes].copy()
    print(f"Dropped {len(genes_to_drop)} flagged genes. Retained {adata.n_vars} clean genes for capacity scoring.")

    # 3. Define Math Operations
    math_ops = {
        'min': lambda x: np.min(x, axis=0),
        'max': lambda x: np.max(x, axis=0),
        'mean': lambda x: np.mean(x, axis=0),
        'sum': lambda x: np.sum(x, axis=0)
    }

    if and_strategy not in math_ops or or_strategy not in math_ops:
        raise ValueError("Strategies must be one of: 'min', 'max', 'mean', 'sum'")

    func_and = math_ops[and_strategy]
    func_or = math_ops[or_strategy]

    # Convert expression matrix to dense format for fast array math across all cells
    expr_matrix = adata.X.toarray() if sp.issparse(adata.X) else adata.X
    gene_expr_dict = {gene: expr_matrix[:, i] for i, gene in enumerate(adata.var_names)}

    # 4. Helper Function: Evaluate GPR Logic
    def evaluate_gpr_array(rule_string):
        """
        Parses a basic DNF (Disjunctive Normal Form) GPR string.
        """
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

    for rxn in model.reactions:
        if rxn.gene_reaction_rule:
            rcs_array = evaluate_gpr_array(rxn.gene_reaction_rule)
            if np.any(rcs_array > 0):
                reaction_ids.append(rxn.id)
                reaction_matrix.append(rcs_array)

    # 6. Build New AnnData Object
    rcs_matrix = np.vstack(reaction_matrix).T

    rcs_adata = AnnData(X=rcs_matrix)
    rcs_adata.obs = adata.obs.copy()
    rcs_adata.var_names = reaction_ids

    print(f"=== RCS Matrix Complete ===")
    print(f"Final shape: {rcs_adata.n_obs} observations x {rcs_adata.n_vars} capable reactions")

    # 7. Save Output
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # FIXED: Tell AnnData to use the backwards-compatible string format
        anndata.settings.allow_write_nullable_strings = False

        rcs_adata.write_h5ad(output_path)
        print(f"Saved RCS matrix to {output_path}")

    return rcs_adata

if __name__ == "__main__":
    pass