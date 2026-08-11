import os
import json
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

def characterize_cluster_metabolism(
    adata_path: str,
    output_dir: str,
    model_json_path: str,
    celltype_column: str = "majority_celltype",
    cluster_column: str = "metabolic_cluster",  # <--- Added parameter here!
    target_reactions: list = None,
    viz_method: str = "umap",
    pval_thresh: float = 0.01,
    lfc_thresh: float = 1.0
):
    """
    Module 2: Characterizes the metabolism of previously clustered cells.
    Loads an annotated AnnData object from a path and generates dot plots, 
    statistical spreadsheets, subsystem enrichment graphs, and multi-reaction feature plots.
    """
    print("=== Initiating Module 2: Metabolic Characterization ===")
    os.makedirs(output_dir, exist_ok=True)
    sc.settings.figdir = output_dir 
    plot_key = viz_method.lower()

    # Load the Annotated AnnData object
    print(f"--> Loading Annotated RCS from {os.path.basename(adata_path)}...")
    adata = sc.read_h5ad(adata_path)
    
    # 🚨 Sanity Check: Ensure the cluster column exists
    if cluster_column not in adata.obs.columns:
        raise ValueError(f"CRITICAL ERROR: '{cluster_column}' not found in adata.obs. Available columns are: {list(adata.obs.columns)}")

    # Load COBRA Model and Map Subsystems
    print(f"--> Extracting subsystem annotations from {os.path.basename(model_json_path)}...")
    with open(model_json_path, 'r') as f:
        model_data = json.load(f)
        
    rxn_to_subsystem = {}
    for rxn in model_data.get('reactions', []):
        sub = rxn.get('subsystem', 'Unknown')
        if isinstance(sub, list):
            sub = sub[0] if sub else 'Unknown'
        rxn_to_subsystem[rxn['id']] = sub if sub else 'Unknown'

    # -------------------------------------------------------------------------
    # 4) FEATURE PLOTS FOR A LIST OF DESIRED REACTIONS
    # -------------------------------------------------------------------------
    if target_reactions:
        print(f"--> Generating feature plots for {len(target_reactions)} target reactions...")
        for rxn in target_reactions:
            if rxn in adata.var_names:
                sc.pl.embedding(
                    adata, basis=plot_key, color=rxn, 
                    cmap='viridis', frameon=False, 
                    show=False, save=f'_{rxn}_capacity.pdf'
                )
            else:
                print(f"WARNING: Target reaction '{rxn}' not found in matrix features.")

    # -------------------------------------------------------------------------
    # 2 & 3) SUBSYSTEM ENRICHMENT GRAPHS & STATISTICAL SPREADSHEETS (CLUSTERS)
    # -------------------------------------------------------------------------
    print(f"--> Identifying enriched reactions and generating subsystem enrichment graphs for '{cluster_column}'...")
    sc.tl.rank_genes_groups(adata, groupby=cluster_column, method='wilcoxon')
    
    cluster_subsystem_dict = {}
    all_sig_cluster_rxns = pd.DataFrame()
    
    for cluster in adata.obs[cluster_column].cat.categories:
        df = sc.get.rank_genes_groups_df(adata, group=cluster)
        sig_df = df[(df['pvals_adj'] < pval_thresh) & (df['logfoldchanges'] > lfc_thresh)].copy()
        sig_df['cluster'] = cluster
        
        # Map Subsystems
        sig_df['subsystem'] = sig_df['names'].map(rxn_to_subsystem).fillna('Unknown')
        all_sig_cluster_rxns = pd.concat([all_sig_cluster_rxns, sig_df])
        
        top_20_sig = sig_df.head(20)
        subsystems = top_20_sig[top_20_sig['subsystem'] != 'Unknown']['subsystem'].unique().tolist()
        cluster_subsystem_dict[f"Cluster_{cluster}"] = subsystems

        # Clean Horizontal Bar Chart of Subsystems driven by top reactions
        clean_sig_df = sig_df[sig_df['subsystem'] != 'Unknown'].copy()
        if not clean_sig_df.empty:
            idx_max_lfc = clean_sig_df.groupby('subsystem')['logfoldchanges'].idxmax()
            top_subs_df = clean_sig_df.loc[idx_max_lfc].sort_values(by='logfoldchanges', ascending=False).head(10)
            
            # Calculate color intensity based on Adjusted P-value
            top_subs_df['neg_log10_padj'] = -np.log10(top_subs_df['pvals_adj'].clip(lower=1e-300))
            
            # Truncated Colormap Integration (0.25 to 0.7 range)
            base_cmap = plt.get_cmap('Oranges')
            cmap = mcolors.LinearSegmentedColormap.from_list(
                'LighterOranges', base_cmap(np.linspace(0.25, 0.7, 256))
            )
            
            norm = plt.Normalize(vmin=top_subs_df['neg_log10_padj'].min() * 0.8, 
                                 vmax=top_subs_df['neg_log10_padj'].max())
            bar_colors = [cmap(norm(val)) for val in top_subs_df['neg_log10_padj']]
            
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
            ax.set_yticks([])
            
            # Overlay text
            for i, (subsystem, lfc) in enumerate(zip(top_subs_df['subsystem'], top_subs_df['logfoldchanges'])):
                ax.text(
                    0.05, i, subsystem, 
                    color='black',
                    va='center', ha='left', 
                    fontsize=11, fontweight='bold'
                )

            # Colorbar Legend
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, pad=0.02)
            cbar.set_label('-log10(Adjusted P-Value)', rotation=270, labelpad=15)

            sns.despine(left=True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"subsystem_enrichment_Cluster_{cluster}.pdf"), dpi=300)
            plt.close()

    # Save cluster spreadsheets 
    all_sig_cluster_rxns.to_csv(os.path.join(output_dir, "filtered_enriched_reactions_by_cluster.csv"), index=False)
    with open(os.path.join(output_dir, "top20_subsystems_by_cluster.json"), 'w') as f:
        json.dump(cluster_subsystem_dict, f, indent=4)

    # -------------------------------------------------------------------------
    # 1) DOT PLOTS FOR TOP ENRICHED REACTIONS (CLUSTERS)
    # -------------------------------------------------------------------------
    print("--> Generating dot plots for cluster-enriched reactions...")
    sc.pl.rank_genes_groups_dotplot(adata, n_genes=5, show=False, save='_cluster_enriched_reactions.pdf')

    # -------------------------------------------------------------------------
    # 1 & 2) DOT PLOTS & STATISTICAL SPREADSHEETS (CELL TYPES)
    # -------------------------------------------------------------------------
    print(f"--> Identifying enriched reactions per cell type ('{celltype_column}')...")
    sc.tl.rank_genes_groups(adata, groupby=celltype_column, method='wilcoxon', key_added='rank_genes_celltype')
    
    all_sig_celltype_rxns = pd.DataFrame()
    for cell_type in adata.obs[celltype_column].cat.categories:
        df = sc.get.rank_genes_groups_df(adata, group=cell_type, key='rank_genes_celltype')
        sig_df = df[(df['pvals_adj'] < pval_thresh) & (df['logfoldchanges'] > lfc_thresh)].copy()
        sig_df['cell_type'] = cell_type
        all_sig_celltype_rxns = pd.concat([all_sig_celltype_rxns, sig_df])
        
    all_sig_celltype_rxns.to_csv(os.path.join(output_dir, "filtered_enriched_reactions_by_celltype.csv"), index=False)
    
    print("--> Generating dot plots for cell type-enriched reactions...")
    sc.pl.rank_genes_groups_dotplot(adata, n_genes=5, key='rank_genes_celltype', show=False, save='_celltype_enriched_reactions.pdf')

    print(f"=== Module 2 Complete. All characterization outputs saved to: {output_dir} ===")
    return adata