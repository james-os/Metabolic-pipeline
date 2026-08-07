import os
import leidenalg
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

def analyze_rcs_matrix(
    adata_path: str,
    output_dir: str,
    celltype_column: str = "cell_type_final",
    target_cell_types: list = None,
    viz_method: str = "umap"
):
    """
    Performs clustering, statistical enrichment, and visualization on an RCS matrix.
    
    Parameters:
    - adata_path (str): Path to the input RCS .h5ad file.
    - output_dir (str): Directory to save publication figures and CSVs.
    - celltype_column (str): The column in adata.obs containing cell type labels.
    - target_cell_types (list): Specific cell types to highlight in final plots.
    - viz_method (str): 'umap' or 'tsne' for dimensionality reduction.
    """
    
    # 0. Environment Setup
    print("=== Initiating RCS Analysis Pipeline ===")
    os.makedirs(output_dir, exist_ok=True)
    adata = sc.read_h5ad(adata_path)
    
    if celltype_column not in adata.obs.columns:
        raise ValueError(f"Column '{celltype_column}' not found in adata.obs.")
        
    viz_method = viz_method.lower()
    if viz_method not in ['umap', 'tsne']:
        raise ValueError("viz_method must be either 'umap' or 'tsne'.")

    # 1. Dimensionality Reduction & Clustering
    print(f"--> Computing PCA, Neighborhood Graph, and {viz_method.upper()}...")
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, key_added='metabolic_cluster')
    
    if viz_method == 'umap':
        sc.tl.umap(adata)
        plot_key = 'umap'
    else:
        sc.tl.tsne(adata)
        plot_key = 'tsne'

    # 1 & 1.5. Produce UMAP/tSNE of Clusters and Cell Types
    print("--> Generating baseline visualisations...")
    sc.settings.figdir = output_dir 
    
    # Plot Clusters
    sc.pl.embedding(
        adata, basis=plot_key, color='metabolic_cluster', 
        frameon=False, title='', legend_loc='on data', 
        show=False, save=f'_metabolic_clusters.pdf'
    )
    
    # Plot All Cell Types
    sc.pl.embedding(
        adata, basis=plot_key, color=celltype_column, 
        frameon=False, title='', 
        show=False, save=f'_all_celltypes.pdf'
    )

    # 2. Pearson Residuals (Statistical Cell Type Enrichment in Clusters)
    print("--> Calculating Pearson Residuals for Cell Type Overrepresentation...")
    # Create contingency table
    observed = pd.crosstab(adata.obs['metabolic_cluster'], adata.obs[celltype_column])
    chi2, p, dof, expected = chi2_contingency(observed)
    
    # Calculate Pearson Residuals: (Observed - Expected) / sqrt(Expected)
    residuals = (observed - expected) / np.sqrt(expected)
    residuals.to_csv(os.path.join(output_dir, "pearson_residuals_clusters_vs_celltypes.csv"))
    
    # Plot and save heatmap of residuals
    plt.figure(figsize=(10, 8))
    
    # Calculate a symmetric limit so white is exactly 0
    limit = np.max(np.abs(residuals.values))
    
    plt.imshow(residuals, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
    plt.colorbar(label='Pearson Residual')
    plt.xticks(ticks=np.arange(len(residuals.columns)), labels=residuals.columns, rotation=90)
    plt.yticks(ticks=np.arange(len(residuals.index)), labels=residuals.index)
    plt.title("Cell Type Enrichment in Metabolic Clusters (Pearson Residuals)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pearson_residuals_heatmap.pdf"), dpi=300)
    plt.close()

    # 3. Differentially Enriched Reactions by Cluster
    print("--> Identifying differentially enriched reactions per metabolic cluster...")
    sc.tl.rank_genes_groups(adata, groupby='metabolic_cluster', method='wilcoxon')
    
    # Save Cluster enrichment to CSV
    cluster_markers = pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(50)
    cluster_markers.to_csv(os.path.join(output_dir, "top50_enriched_reactions_by_cluster.csv"))
    
    # Plot Dotplot for Clusters
    sc.pl.rank_genes_groups_dotplot(
        adata, n_genes=5, 
        show=False, save=f'_cluster_enriched_reactions.pdf'
    )

    # 4. Differentially Enriched Reactions by Cell Type
    print("--> Identifying differentially enriched reactions per biological cell type...")
    sc.tl.rank_genes_groups(adata, groupby=celltype_column, method='wilcoxon', key_added='rank_genes_celltype')
    
    # Save Cell Type enrichment to CSV
    celltype_markers = pd.DataFrame(adata.uns['rank_genes_celltype']['names']).head(50)
    celltype_markers.to_csv(os.path.join(output_dir, "top50_enriched_reactions_by_celltype.csv"))
    
    # Plot Dotplot for Cell Types
    sc.pl.rank_genes_groups_dotplot(
        adata, n_genes=5, key='rank_genes_celltype',
        show=False, save=f'_celltype_enriched_reactions.pdf'
    )

    # 5. Highlight Specific Target Cell Types
    if target_cell_types:
        print(f"--> Highlighting target populations: {target_cell_types}")
        sc.pl.embedding(
            adata, 
            basis=plot_key, 
            color=celltype_column, 
            groups=target_cell_types,
            na_color='lightgrey', 
            frameon=False, 
            title='', 
            show=False, 
            save=f'_highlighted_targets.pdf'
        )

    print(f"=== Analysis Complete. All outputs saved to: {output_dir} ===")
    return adata