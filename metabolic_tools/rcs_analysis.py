import os
import json
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

def analyze_rcs_matrix(
    adata_path: str,
    output_dir: str,
    model_json_path: str,
    celltype_column: str = "majority_celltype",
    target_cell_types: list = None,
    viz_method: str = "umap",
    pval_thresh: float = 0.01,
    lfc_thresh: float = 1.0
):
    """
    Performs clustering, statistical enrichment, subsystem mapping, and visualization.
    """
    
    print("=== Initiating Upgraded RCS Analysis Pipeline ===")
    os.makedirs(output_dir, exist_ok=True)
    adata = sc.read_h5ad(adata_path)
    
    # 0. Load COBRA Model and Map Subsystems
    print(f"--> Extracting subsystem annotations from {os.path.basename(model_json_path)}...")
    with open(model_json_path, 'r') as f:
        model_data = json.load(f)
        
    rxn_to_subsystem = {}
    for rxn in model_data.get('reactions', []):
        sub = rxn.get('subsystem', 'Unknown')
        # Handle cases where subsystem is stored as a list of strings
        if isinstance(sub, list):
            sub = sub[0] if sub else 'Unknown'
        # Fallback for empty strings
        if not sub:
            sub = 'Unknown'
        rxn_to_subsystem[rxn['id']] = sub

    # 1. Dimensionality Reduction & Clustering
    print(f"--> Computing PCA, Neighborhood Graph, and {viz_method.upper()}...")
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, key_added='metabolic_cluster')
    
    if viz_method.lower() == 'umap':
        sc.tl.umap(adata)
        plot_key = 'umap'
    else:
        sc.tl.tsne(adata)
        plot_key = 'tsne'

    # 1.5 Produce Baseline Visualizations
    sc.settings.figdir = output_dir 
    sc.pl.embedding(adata, basis=plot_key, color='metabolic_cluster', frameon=False, show=False, save='_metabolic_clusters.pdf')
    sc.pl.embedding(adata, basis=plot_key, color=celltype_column, frameon=False, show=False, save='_all_celltypes.pdf')

    # 2. Pearson Residuals with Dendrogram (Seaborn Clustermap)
    print("--> Calculating Pearson Residuals and generating Clustermap...")
    observed = pd.crosstab(adata.obs['metabolic_cluster'], adata.obs[celltype_column])
    chi2, p, dof, expected = chi2_contingency(observed)
    residuals = (observed - expected) / np.sqrt(expected)
    residuals.to_csv(os.path.join(output_dir, "pearson_residuals.csv"))
    
    # Seaborn clustermap automatically calculates hierarchical clustering (dendrogram)
    cg = sns.clustermap(
        residuals, 
        cmap='RdBu_r', 
        center=0, 
        figsize=(12, 10),
        cbar_kws={'label': 'Pearson Residual'},
        dendrogram_ratio=(0.1, 0.2)
    )
    plt.setp(cg.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
    cg.savefig(os.path.join(output_dir, "pearson_residuals_clustermap.pdf"), dpi=300)
    plt.close()

    # 3. Differentially Enriched Reactions by Cluster & Subsystem Mapping
    print("--> Identifying enriched reactions and subsystems per cluster...")
    sc.tl.rank_genes_groups(adata, groupby='metabolic_cluster', method='wilcoxon')
    
    cluster_subsystem_dict = {}
    all_sig_cluster_rxns = pd.DataFrame()
    
    for cluster in adata.obs['metabolic_cluster'].cat.categories:
        # Extract full dataframe for the cluster
        df = sc.get.rank_genes_groups_df(adata, group=cluster)
        
        # Apply statistical thresholds
        sig_df = df[(df['pvals_adj'] < pval_thresh) & (df['logfoldchanges'] > lfc_thresh)].copy()
        sig_df['cluster'] = cluster
        all_sig_cluster_rxns = pd.concat([all_sig_cluster_rxns, sig_df])
        
        # Extract subsystems for the top 20 significant hits
        top_20_sig = sig_df.head(20)
        subsystems = top_20_sig['names'].map(rxn_to_subsystem).dropna().unique().tolist()
        
        # Filter out 'Unknown' if you prefer clean data
        subsystems = [s for s in subsystems if s != 'Unknown']
        cluster_subsystem_dict[f"Cluster_{cluster}"] = subsystems

    # Save filtered cluster reactions and the subsystem mapping
    all_sig_cluster_rxns.to_csv(os.path.join(output_dir, "filtered_enriched_reactions_by_cluster.csv"), index=False)
    with open(os.path.join(output_dir, "top20_subsystems_by_cluster.json"), 'w') as f:
        json.dump(cluster_subsystem_dict, f, indent=4)

    sc.pl.rank_genes_groups_dotplot(adata, n_genes=5, show=False, save='_cluster_enriched_reactions.pdf')

    # 4. Differentially Enriched Reactions by Cell Type
    print("--> Identifying enriched reactions per cell type...")
    sc.tl.rank_genes_groups(adata, groupby=celltype_column, method='wilcoxon', key_added='rank_genes_celltype')
    
    all_sig_celltype_rxns = pd.DataFrame()
    for cell_type in adata.obs[celltype_column].cat.categories:
        df = sc.get.rank_genes_groups_df(adata, group=cell_type, key='rank_genes_celltype')
        sig_df = df[(df['pvals_adj'] < pval_thresh) & (df['logfoldchanges'] > lfc_thresh)].copy()
        sig_df['cell_type'] = cell_type
        all_sig_celltype_rxns = pd.concat([all_sig_celltype_rxns, sig_df])
        
    all_sig_celltype_rxns.to_csv(os.path.join(output_dir, "filtered_enriched_reactions_by_celltype.csv"), index=False)
    sc.pl.rank_genes_groups_dotplot(adata, n_genes=5, key='rank_genes_celltype', show=False, save='_celltype_enriched_reactions.pdf')

    # 5. Highlight Specific Target Cell Types
    if target_cell_types:
        sc.pl.embedding(adata, basis=plot_key, color=celltype_column, groups=target_cell_types, na_color='lightgrey', frameon=False, show=False, save='_highlighted_targets.pdf')

    print(f"=== Analysis Complete. All outputs saved to: {output_dir} ===")
    return adata