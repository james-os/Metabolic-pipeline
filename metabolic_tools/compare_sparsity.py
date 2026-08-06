import scanpy as sc
import cobra
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
import os

def compare_metabolic_sparsity(
    sc_h5ad_path, 
    mc_h5ad_path, 
    model_path,           # Accepts .xml, .sbml, or .json
    plot_save_path=None   
):
    """
    Compares the dropout rate of metabolic genes between a single-cell 
    and metacell matrix, printing statistics and optionally saving a violin plot.
    """
    print("1. Loading datasets and model...")
    adata_sc = sc.read_h5ad(sc_h5ad_path)
    adata_mc = sc.read_h5ad(mc_h5ad_path)
    
    # Dynamically load the model based on file extension
    if model_path.endswith('.json'):
        model = cobra.io.load_json_model(model_path)
    elif model_path.endswith('.xml') or model_path.endswith('.sbml'):
        model = cobra.io.read_sbml_model(model_path)
    else:
        raise ValueError("Unsupported model format. Please provide a .json, .xml, or .sbml file.")
    
    # 2. Isolate Metabolic Genes
    model_genes = [g.id for g in model.genes]
    shared_genes_sc = list(set(model_genes).intersection(adata_sc.var_names))
    shared_genes_mc = list(set(model_genes).intersection(adata_mc.var_names))
    
    consensus_genes = list(set(shared_genes_sc).intersection(shared_genes_mc))
    print(f"   Found {len(consensus_genes)} metabolic genes shared across model and datasets.")
    
    adata_sc_met = adata_sc[:, consensus_genes]
    adata_mc_met = adata_mc[:, consensus_genes]
    
    # 3. Define Sparsity Calculation Helper
    def calculate_gene_dropout(X):
        if sp.issparse(X):
            non_zeros_per_column = X.getnnz(axis=0)
            return 1.0 - (non_zeros_per_column / X.shape[0])
        else:
            non_zeros_per_column = np.count_nonzero(X, axis=0)
            return 1.0 - (non_zeros_per_column / X.shape[0])

    print("2. Calculating sparsity matrices...")
    sc_dropout_rates = calculate_gene_dropout(adata_sc_met.X)
    mc_dropout_rates = calculate_gene_dropout(adata_mc_met.X)
    
    # 4. Quantification & Statistics
    sc_q1, sc_med, sc_q3 = np.percentile(sc_dropout_rates, [25, 50, 75])
    mc_q1, mc_med, mc_q3 = np.percentile(mc_dropout_rates, [25, 50, 75])
    
    print("\n--- GLOBAL METABOLIC SPARSITY ---")
    print(f"Single-Cell Average Dropout: {np.mean(sc_dropout_rates)*100:.2f}%")
    print(f"Metacell Average Dropout:    {np.mean(mc_dropout_rates)*100:.2f}%")
    
    print("\n--- DETAILED DROPOUT QUANTIFICATION ---")
    print("Single-Cell Distribution:")
    print(f"  Median: {sc_med*100:.2f}% (Q1: {sc_q1*100:.2f}%, Q3: {sc_q3*100:.2f}%)")
    print("Metacell Distribution:")
    print(f"  Median: {mc_med*100:.2f}% (Q1: {mc_q1*100:.2f}%, Q3: {mc_q3*100:.2f}%)")
    
    # 5. Prepare Data for Plotting
    df_sc = pd.DataFrame({'Gene': consensus_genes, 'Dropout_Fraction': sc_dropout_rates, 'Dataset': 'Single Cell'})
    df_mc = pd.DataFrame({'Gene': consensus_genes, 'Dropout_Fraction': mc_dropout_rates, 'Dataset': 'Metacell'})
    df_plot = pd.concat([df_sc, df_mc], ignore_index=True)
    
    # 6. Generate Violin Plot
    print("\n3. Generating visualization...")
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=df_plot, 
        x='Dataset', 
        y='Dropout_Fraction', 
        palette={'Single Cell': '#E66100', 'Metacell': '#5D3A9B'},
        inner='quartile'
    )
    plt.title("Reduction of Metabolic Gene Dropout via Metacell Aggregation", fontsize=14, fontweight='bold')
    plt.ylabel("Fraction of Cells/Metacells with 0.0 Expression", fontsize=12)
    plt.xlabel("")
    plt.ylim(-0.05, 1.05)
    plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # File Saving Logic
    if plot_save_path:
        os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
        plt.savefig(plot_save_path, bbox_inches='tight', dpi=300)
        print(f"   Plot successfully saved to: {plot_save_path}")
        
    plt.show()
