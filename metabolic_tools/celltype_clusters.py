import os
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe 
import seaborn as sns  
from scipy.stats import chi2_contingency, norm
from statsmodels.stats.multitest import multipletests 

def generate_annotated_rcs_state(
    adata_path,
    output_dir,
    celltype_col="majority_celltype",
    cluster_resolution=1.0,
    alpha=0.05,        
    embedding="umap",  
    point_size=80,
    highlight_celltypes=None,
    label_size=10      # <--- NEW: Controls the font size of the UMAP labels
):
    """
    Module 1: The State Generator.
    Loads an RCS matrix, computes rapid igraph clustering, dynamically names 
    archetypes using FDR-corrected Pearson residuals, and outputs comprehensive 
    visualizations and a statistical summary spreadsheet.
    """
    # 1. Setup and Load
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading RCS matrix from {adata_path}...")
    adata = sc.read_h5ad(adata_path)
    
    # 2. Dimensionality Reduction & Rapid Clustering
    print("Computing PCA, Graph, and Leiden clustering...")
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=40)
    sc.tl.leiden(adata, resolution=cluster_resolution, key_added='leiden', flavor='igraph', n_iterations=2, directed=False)
    
    embedding = embedding.lower()
    if embedding in ['umap', 'both']: sc.tl.umap(adata)
    if embedding in ['tsne', 'both']: sc.tl.tsne(adata) 

    # 3. Base Statistics & FDR Calculations
    print(f"Calculating Pearson residuals and FDR for leiden vs {celltype_col}...")
    cluster_counts = pd.crosstab(adata.obs['leiden'], adata.obs[celltype_col])
    cluster_props = cluster_counts.div(cluster_counts.sum(axis=1), axis=0) * 100
    
    chi2, p, dof, expected = chi2_contingency(cluster_counts)
    residuals = pd.DataFrame((cluster_counts.values - expected) / np.sqrt(expected), index=cluster_counts.index, columns=cluster_counts.columns)
    
    p_vals = norm.sf(residuals.values) 
    _, p_adj_flat, _, _ = multipletests(p_vals.flatten(), alpha=alpha, method='fdr_bh')
    p_adj = pd.DataFrame(p_adj_flat.reshape(residuals.shape), index=residuals.index, columns=residuals.columns)
    
    # 4. Data-Driven Cluster Naming
    print("Dynamically generating metabolic archetypes...")
    cluster_to_archetype = {}
    cluster_to_umap_label = {}
    
    for cluster in residuals.index:
        enriched_mask = (p_adj.loc[cluster] < alpha) & (residuals.loc[cluster] > 0)
        enriched_cts = residuals.loc[cluster][enriched_mask].sort_values(ascending=False)
        
        if enriched_cts.empty:
            archetype_name = "Mixed Epithelial"
            label = f"Mixed\n(Non-Specific)"
        else:
            top_cts = enriched_cts.head(4).index.tolist()
            archetype_name = ", ".join(top_cts)
            total_prop = cluster_props.loc[cluster, top_cts].sum()
            label = f"{archetype_name}\n({total_prop:.0f}%)"
            
        cluster_to_archetype[cluster] = archetype_name
        cluster_to_umap_label[cluster] = label

    adata.obs['metabolic_archetype'] = adata.obs['leiden'].map(cluster_to_archetype).astype('category')
    adata.obs['embedding_label'] = adata.obs['leiden'].map(cluster_to_umap_label).astype('category')
    
    # 5. GENERATE STATISTICAL SPREADSHEET
    print("Exporting Statistical Summary...")
    stats_list = []
    for cluster in residuals.index:
        sig_mask = (p_adj.loc[cluster] < alpha) & (residuals.loc[cluster] > 0)
        sig_types = residuals.columns[sig_mask].tolist()
        
        for ct in sig_types:
            stats_list.append({
                'Leiden_Cluster': cluster,
                'Archetype_Name': cluster_to_archetype[cluster],
                'Cell_Type': ct,
                'Z_Score_Pearson': residuals.loc[cluster, ct],
                'FDR_P_Value': p_adj.loc[cluster, ct],
                'Cluster_Composition_Pct': cluster_props.loc[cluster, ct]
            })
            
    stats_df = pd.DataFrame(stats_list).sort_values(by=['Leiden_Cluster', 'Z_Score_Pearson'], ascending=[True, False])
    stats_df.to_csv(f"{output_dir}/Statistical_Overrepresentation_Summary.csv", index=False)

    # ---------------------------------------------------------
    # 6. VISUALIZATIONS 
    # ---------------------------------------------------------
    plot_basis = 'umap' if embedding in ['umap', 'both'] else 'tsne'
    plot_func = sc.pl.umap if plot_basis == 'umap' else sc.pl.tsne
    
    # A. Annotated Archetype Embedding
    print("Plotting UMAPs...")
    fig_emb, ax_emb = plt.subplots(figsize=(9, 9))
    plot_func(adata, color='embedding_label', size=point_size, legend_loc='on data', 
              legend_fontsize=label_size, palette='Set2', ax=ax_emb, show=False, frameon=False, 
              title=f'Metabolic Archetypes ({plot_basis.upper()})')
              
    # <--- INCREASED HALO THICKNESS AND OPACITY HERE
    for text in ax_emb.texts: 
        text.set_path_effects([pe.withStroke(linewidth=4.5, foreground=(1.0, 1.0, 1.0, 0.9))])
        
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{plot_basis.upper()}_Metabolic_Archetypes.pdf", bbox_inches='tight', transparent=True)
    plt.close(fig_emb)

    # B1. Original Cell Types Embedding 
    print("Plotting Original Cell Types...")
    fig_ct, ax_ct = plt.subplots(figsize=(10, 8)) 
    plot_func(adata, color=celltype_col, size=point_size, palette='tab20', 
              ax=ax_ct, show=False, frameon=False, 
              title=f'Original Cell Types ({plot_basis.upper()})')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{plot_basis.upper()}_Original_CellTypes.pdf", bbox_inches='tight', transparent=True)
    plt.close(fig_ct)
    
    # B2. Targeted Highlight Cell Types 
    if highlight_celltypes is not None:
        if isinstance(highlight_celltypes, str):
            highlight_celltypes = [highlight_celltypes] 
            
        print(f"Plotting Highlighted Cell Types: {highlight_celltypes}...")
        fig_hl, ax_hl = plt.subplots(figsize=(10, 8)) 
        plot_func(adata, color=celltype_col, size=point_size, groups=highlight_celltypes,
                  palette='tab10', ax=ax_hl, show=False, frameon=False, 
                  title=f'Targeted Highlight: {", ".join(highlight_celltypes)}')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{plot_basis.upper()}_Highlighted_CellTypes.pdf", bbox_inches='tight', transparent=True)
        plt.close(fig_hl)

    # C. Stacked Bar Chart (Percentages)
    print("Plotting Bar Charts...")
    fig_bar1, ax_bar1 = plt.subplots(figsize=(8, 6))
    cluster_props.plot(kind='bar', stacked=True, ax=ax_bar1, colormap='tab20', edgecolor='white', linewidth=0.5)
    ax_bar1.set_title('Cluster Composition (Percentages)', pad=15)
    ax_bar1.set_xlabel('Leiden Cluster', labelpad=10)
    ax_bar1.set_ylabel('Percentage of Cells (%)')
    ax_bar1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, title="Cell Types")
    ax_bar1.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/BarChart_Composition_Percentage.pdf", bbox_inches='tight', transparent=True)
    plt.close(fig_bar1)

    # D. Stacked Bar Chart (Absolute Counts)
    fig_bar2, ax_bar2 = plt.subplots(figsize=(8, 6))
    cluster_counts.plot(kind='bar', stacked=True, ax=ax_bar2, colormap='tab20', edgecolor='white', linewidth=0.5)
    ax_bar2.set_title('Cluster Composition (Absolute Cells)', pad=15)
    ax_bar2.set_xlabel('Leiden Cluster', labelpad=10)
    ax_bar2.set_ylabel('Total Number of Cells')
    ax_bar2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, title="Cell Types")
    ax_bar2.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/BarChart_Composition_Absolute.pdf", bbox_inches='tight', transparent=True)
    plt.close(fig_bar2)

    # E. Custom Dotplot (Cell Types vs Clusters)
    print("Plotting Custom Relationship Dotplot...")
    dot_data = pd.DataFrame({
        'Cluster': np.repeat(residuals.index, len(residuals.columns)),
        'Cell Type': np.tile(residuals.columns, len(residuals.index)),
        'Percentage': cluster_props.values.flatten(),
        'Residual': residuals.values.flatten()
    })
    
    max_abs_res = max(abs(dot_data['Residual'].min()), abs(dot_data['Residual'].max()))
    
    fig_dot, ax_dot = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        data=dot_data, x='Cluster', y='Cell Type',
        size='Percentage', sizes=(10, 800),
        hue='Residual', palette='RdBu_r', hue_norm=(-max_abs_res, max_abs_res),
        ax=ax_dot, edgecolor='gray', linewidth=0.5
    )
    ax_dot.set_title('Cell Type Representation Across Clusters', pad=20)
    ax_dot.legend(bbox_to_anchor=(1.05, 1), loc='upper left', labelspacing=1.5)
    ax_dot.grid(True, linestyle='--', alpha=0.3) 
    plt.tight_layout()
    plt.savefig(f"{output_dir}/DotPlot_Cluster_Relationships.pdf", bbox_inches='tight', transparent=True)
    plt.close(fig_dot)

    # F. Pearson Residual Clustermap
    print("Plotting Pearson Residual Clustermap...")
    cmap_fig = sns.clustermap(
        residuals, cmap='RdBu_r', center=0, figsize=(12, 10),
        yticklabels=True, xticklabels=True,
        cbar_pos=(0.02, 0.8, 0.05, 0.18), cbar_kws={'label': 'Pearson Residual (Z-score)'}
    )
    cmap_fig.ax_heatmap.set_xlabel("True Cell Type")
    cmap_fig.ax_heatmap.set_ylabel("Leiden Cluster")
    plt.setp(cmap_fig.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
    cmap_fig.savefig(f"{output_dir}/Heatmap_Pearson_Residuals.pdf", bbox_inches='tight')
    plt.close(cmap_fig.fig)

    # 7. Save the State
    save_path = f"{output_dir}/Annotated_RCS.h5ad"
    print(f"Saving fully annotated state to {save_path}...")
    adata.write(save_path)
    print("Module 1 Complete!")
    return save_path