import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import json 

def characterise_metabolism(
    adata_path: str,
    output_dir: str,
    celltype_column: str = "majority_celltype",
    cluster_column: str = "metabolic_cluster",
    target_reactions: list = None,
    viz_method: str = "umap",
    pval_thresh: float = 0.01,
    lfc_thresh: float = 1.0
):
    """
    Module 2: Characterizes the metabolism of previously clustered cells.
    Loads an annotated AnnData object from a path and generates dot plots, 
    statistical spreadsheets, subsystem enrichment graphs, and multi-reaction feature plots.
    Subsystem metadata is extracted directly from the AnnData object.
    """
    print("=== Initiating Module 2: Metabolic Characterization ===")
    os.makedirs(output_dir, exist_ok=True)
    sc.settings.figdir = output_dir 
    plot_key = viz_method.lower()

    # Load the Annotated AnnData object
    print(f"--> Loading Annotated RCS from {os.path.basename(adata_path)}...")
    adata = sc.read_h5ad(adata_path)
    
    # Sanity Check: Ensure the cluster column exists
    if cluster_column not in adata.obs.columns:
        raise ValueError(f"CRITICAL ERROR: '{cluster_column}' not found in adata.obs. Available columns are: {list(adata.obs.columns)}")

    # Sanity Check: Ensure subsystem metadata exists from the ECS calculator
    if 'subsystem' not in adata.var.columns:
        print("WARNING: 'subsystem' column not found in adata.var. Subsystems will be labeled as 'Unknown'.")
        adata.var['subsystem'] = 'Unknown'

    # -------------------------------------------------------------------------
    # 4) FEATURE PLOTS FOR SUBSTRING SEARCHED REACTIONS
    # -------------------------------------------------------------------------
    if target_reactions:
        print(f"--> Generating feature plots. Searching for {len(target_reactions)} target strings...")
        
        # Step A: Collect all unique matching reactions across all target strings
        all_matching_rxns = set()
        for target_str in target_reactions:
            matching_rxns = [var for var in adata.var_names if target_str in var]
            if matching_rxns:
                all_matching_rxns.update(matching_rxns)
            else:
                print(f"WARNING: No features found containing the target string '{target_str}'.")
                
        if all_matching_rxns:
            all_matching_rxns = list(all_matching_rxns)
            
            # Step B: Calculate the global maximum capacity score across all matches
            expr_data = adata[:, all_matching_rxns].X
            global_max = expr_data.max()
            
            # Step C: Build the custom Magma colormap with a light grey absolute zero
            magma_colors = plt.get_cmap('viridis')(np.linspace(0, 1, 256))
            magma_colors[0] = mcolors.to_rgba('lightgrey')
            custom_magma = mcolors.LinearSegmentedColormap.from_list('magma_grey_zero', magma_colors)
            
            print(f"--> Found {len(all_matching_rxns)} unique reactions. Generating plots with shared max value: {global_max:.2f}")
            
            # Step D: Plot each reaction using the shared scale and custom colormap
            for rxn in all_matching_rxns:
                # Clean the reaction name for safe file saving
                safe_rxn_name = str(rxn).replace('/', '_').replace('\\', '_').replace(':', '_')
                
                # Generate the plot and capture the Axis object
                ax = sc.pl.embedding(
                    adata, basis=plot_key, color=rxn, 
                    size=150,         
                    cmap=custom_magma, vmin=0, vmax=global_max, frameon=False, 
                    show=False
                )
                
                # Save using standard Matplotlib to avoid the Scanpy deprecation warning
                save_path = os.path.join(output_dir, f"{plot_key}_{safe_rxn_name}_capacity.png")
                ax.figure.savefig(save_path, dpi=300, bbox_inches='tight')
                
                # Close the figure to prevent memory leaks from opening too many plots
                plt.close(ax.figure)

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
        
        # Map Subsystems directly from adata.var
        sig_df['subsystem'] = sig_df['names'].map(adata.var['subsystem']).fillna('Unknown')
        all_sig_cluster_rxns = pd.concat([all_sig_cluster_rxns, sig_df])
        
        top_20_sig = sig_df.head(20)
        subsystems = top_20_sig[top_20_sig['subsystem'] != 'Unknown']['subsystem'].unique().tolist()
        
        # Format a safe string for saving files/keys
        safe_cluster_name = str(cluster).replace('/', '_').replace('\\', '_')
        cluster_subsystem_dict[f"Cluster_{safe_cluster_name}"] = subsystems

        # Clean Horizontal Bar Chart of Subsystems driven by top reactions
        clean_sig_df = sig_df[sig_df['subsystem'] != 'Unknown'].copy()
        if not clean_sig_df.empty:
            # Sort by logfoldchanges descending to easily grab top reactions
            clean_sig_df_sorted = clean_sig_df.sort_values(by='logfoldchanges', ascending=False)
            
            idx_max_lfc = clean_sig_df_sorted.groupby('subsystem')['logfoldchanges'].idxmax()
            top_subs_df = clean_sig_df_sorted.loc[idx_max_lfc].sort_values(by='logfoldchanges', ascending=False).head(10)
            
            # Create a combined display string for the Y-axis
            display_names = []
            for sub in top_subs_df['subsystem']:
                top_3 = clean_sig_df_sorted[clean_sig_df_sorted['subsystem'] == sub].head(3)['names'].tolist()
                display_names.append(f"{sub} - {', '.join(top_3)}")
            top_subs_df['display_name'] = display_names
            
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
            
            plt.figure(figsize=(11, 6))
            ax = sns.barplot(
                x='logfoldchanges', 
                y='display_name',  # Mapped to the new combined string
                data=top_subs_df, 
                hue='display_name', 
                palette=bar_colors,
                legend=False
            )
            
            plt.xlabel('Max Log-Fold Change of Driving Reaction')
            plt.ylabel('')
            plt.title(f'Top Driven Subsystems (Cluster {cluster})')
            
            # Style the Y-axis labels instead of hiding them
            plt.yticks(fontsize=10, fontweight='bold')

            # Colorbar Legend
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, pad=0.02)
            cbar.set_label('-log10(Adjusted P-Value)', rotation=270, labelpad=15)

            sns.despine(left=False) 
            plt.tight_layout()
            
            # bbox_inches='tight' guarantees text is never cut off
            plt.savefig(os.path.join(output_dir, f"subsystem_enrichment_Cluster_{safe_cluster_name}.png"), dpi=300, bbox_inches='tight')
            plt.close()

    # Save cluster spreadsheets 
    all_sig_cluster_rxns.to_csv(os.path.join(output_dir, "filtered_enriched_reactions_by_cluster.csv"), index=False)
    with open(os.path.join(output_dir, "top20_subsystems_by_cluster.json"), 'w') as f:
        json.dump(cluster_subsystem_dict, f, indent=4)

    # -------------------------------------------------------------------------
    # 1) DOT PLOTS FOR TOP ENRICHED REACTIONS (CLUSTERS)
    # -------------------------------------------------------------------------
    print("--> Generating dot plots for cluster-enriched reactions...")
    sc.pl.rank_genes_groups_dotplot(adata, n_genes=5, show=False, save='_cluster_enriched_reactions.png')

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
    sc.pl.rank_genes_groups_dotplot(adata, n_genes=5, key='rank_genes_celltype', show=False, save='_celltype_enriched_reactions.png')

    print(f"=== Module 2 Complete. All characterization outputs saved to: {output_dir} ===")
    return adata