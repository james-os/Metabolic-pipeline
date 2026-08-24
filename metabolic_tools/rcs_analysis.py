import os
import json
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors  # Required for truncating the colormap
import seaborn as sns
from scipy.stats import chi2_contingency

def analyze_rcs_matrix(
    adata_path: str,
    output_dir: str,
    model_json_path: str,
    celltype_column: str = "majority_celltype",
    target_cell_types: list = None,
    target_reaction: str = None,
    viz_method: str = "umap",
    pval_thresh: float = 0.01,
    lfc_thresh: float = 1.0
):
    """
    Performs clustering, statistical enrichment, subsystem mapping, and visualization 
    on a Reaction Capacity Score (RCS) matrix.
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
        if isinstance(sub, list):
            sub = sub[0] if sub else 'Unknown'
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

    sc.settings.figdir = output_dir 

    # 1.5 Produce Baseline Visualizations & Target Reaction Feature Plot
    print("--> Generating embedding visualizations...")
    sc.pl.embedding(adata, basis=plot_key, color='metabolic_cluster', frameon=False, show=False, save='_metabolic_clusters.pdf')
    sc.pl.embedding(adata, basis=plot_key, color=celltype_column, frameon=False, show=False, save='_all_celltypes.pdf')
    
    # Plot specific reaction capacity intensity
    if target_reaction:
        if target_reaction in adata.var_names:
            print(f"--> Generating feature plot for target reaction: {target_reaction}")
            sc.pl.embedding(
                adata, basis=plot_key, color=target_reaction, 
                cmap='viridis', frameon=False, 
                show=False, save=f'_{target_reaction}_capacity.pdf'
            )
        else:
            print(f"WARNING: Target reaction '{target_reaction}' not found in matrix features.")

    # 2. Pearson Residuals with Dendrogram (Seaborn Clustermap)
    print("--> Calculating Pearson Residuals and generating Clustermap...")
    observed = pd.crosstab(adata.obs['metabolic_cluster'], adata.obs[celltype_column])
    chi2, p, dof, expected = chi2_contingency(observed)
    residuals = (observed - expected) / np.sqrt(expected)
    residuals.to_csv(os.path.join(output_dir, "pearson_residuals.csv"))
    
    cg = sns.clustermap(
        residuals, cmap='RdBu_r', center=0, figsize=(12, 10),
        cbar_kws={'label': 'Pearson Residual'}, dendrogram_ratio=(0.1, 0.2)
    )
    plt.setp(cg.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
    cg.savefig(os.path.join(output_dir, "pearson_residuals_clustermap.pdf"), dpi=300)
    plt.close()

    # 3. Differentially Enriched Reactions by Cluster & Subsystem Bar Charts
    print("--> Identifying enriched reactions and plotting subsystem bar charts per cluster...")
    sc.tl.rank_genes_groups(adata, groupby='metabolic_cluster', method='wilcoxon')
    
    cluster_subsystem_dict = {}
    all_sig_cluster_rxns = pd.DataFrame()
    
    for cluster in adata.obs['metabolic_cluster'].cat.categories:
        df = sc.get.rank_genes_groups_df(adata, group=cluster)
        sig_df = df[(df['pvals_adj'] < pval_thresh) & (df['logfoldchanges'] > lfc_thresh)].copy()
        sig_df['cluster'] = cluster
        
        # Map Subsystems to the significant reactions
        sig_df['subsystem'] = sig_df['names'].map(rxn_to_subsystem).fillna('Unknown')
        all_sig_cluster_rxns = pd.concat([all_sig_cluster_rxns, sig_df])
        
        # Isolate known subsystems for the top 20 list (JSON)
        top_20_sig = sig_df.head(20)
        subsystems = top_20_sig[top_20_sig['subsystem'] != 'Unknown']['subsystem'].unique().tolist()
        cluster_subsystem_dict[f"Cluster_{cluster}"] = subsystems

        # Clean Horizontal Bar Chart of Subsystems driven by top reactions
        clean_sig_df = sig_df[sig_df['subsystem'] != 'Unknown'].copy()
        if not clean_sig_df.empty:
            # Find the index of the specific driving reaction (max LFC) for each subsystem
            idx_max_lfc = clean_sig_df.groupby('subsystem')['logfoldchanges'].idxmax()
            top_subs_df = clean_sig_df.loc[idx_max_lfc].sort_values(by='logfoldchanges', ascending=False).head(10)
            
            # Calculate color intensity based on Adjusted P-value (-log10 for visualization scaling)
            top_subs_df['neg_log10_padj'] = -np.log10(top_subs_df['pvals_adj'].clip(lower=1e-300))
            
            # --- METHOD 1: Truncated Colormap Integration ---
            base_cmap = plt.get_cmap('Oranges')
            # Extract only the 0% to 70% range of the colormap to prevent it from getting too dark
            cmap = mcolors.LinearSegmentedColormap.from_list(
                'LighterOranges', base_cmap(np.linspace(0.25, 0.7, 256))
            )
            
            norm = plt.Normalize(vmin=top_subs_df['neg_log10_padj'].min() * 0.8, 
                                 vmax=top_subs_df['neg_log10_padj'].max())
            bar_colors = [cmap(norm(val)) for val in top_subs_df['neg_log10_padj']]
            # ------------------------------------------------
            
            plt.figure(figsize=(9, 6)) 
            ax = sns.barplot(
                x='logfoldchanges', 
                y='subsystem', 
                data=top_subs_df, 
                hue='subsystem', 
                palette=bar_colors,
                legend=False
            )
            
            plt.xlabel('Max Log-Fold Change of Driving Reaction')
            plt.ylabel('')
            plt.title(f'Top Driven Subsystems (Cluster {cluster})')
            
            # Remove the standard y-axis ticks and labels
            ax.set_yticks([])
            
            # Overlay the subsystem names directly onto the bars
            for i, (subsystem, lfc) in enumerate(zip(top_subs_df['subsystem'], top_subs_df['logfoldchanges'])):
                ax.text(
                    0.05, i, subsystem, 
                    color='black',  # Dark grey text for readability
                    va='center', ha='left', 
                    fontsize=11, fontweight='bold'
                )

            # Generate the Colorbar Legend
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, pad=0.02)
            cbar.set_label('-log10(Adjusted P-Value)', rotation=270, labelpad=15)

            # Clean up the borders
            sns.despine(left=True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"subsystem_enrichment_Cluster_{cluster}.pdf"), dpi=300)
            plt.close()

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