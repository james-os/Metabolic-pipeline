import os
import re
import json
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.transforms as mtransforms
import seaborn as sns

def characterise_metabolism(
    adata_path: str,
    output_dir: str,
    celltype_column: str = "majority_celltype",
    cluster_column: str = "metabolic_cluster",
    target_reactions: list = None,
    viz_method: str = "umap",
    point_size: int = 150,
    pval_thresh: float = 0.01,
    lfc_thresh: float = 1.0
):
    """
    Module 2: Characterizes the metabolism of previously clustered cells.
    Outputs:
      - Feature plots for target reactions
      - Top 15 reactions bar charts grouped by subsystem with right-aligned brackets
      - Natural y-axis limits matching the actual reaction count
      - Tunable colorbar positioning for publication/Inkscape layout
      - Completely italicized gene symbols (including trailing numerals)
      - Filtered statistical CSVs and cluster summary JSONs
      - Scanpy dot plots for clusters and cell types
    """
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.it'] = 'DejaVu Sans:italic'

    def _italicize_token(token: str) -> str:
        clean = token.strip()
        if not clean:
            return ""
        safe = clean.replace('*', r'\ast')
        return f"\\mathit{{{safe}}}"

    def _format_rich_label(rxn_name: str) -> str:
        rxn_str = str(rxn_name).strip()
        
        match = re.search(r'^(.*?)\s*\((.*?)(\s+isozyme)?\)$', rxn_str)
        if match:
            base_rxn = match.group(1).strip()
            gene_part = match.group(2).strip()
            is_isozyme = match.group(3) is not None
            
            if re.search(r'[A-Za-z]', gene_part) and len(gene_part.split()) <= 4:
                tokens = re.split(r'([+/])', gene_part)
                formatted_tokens = []
                for t in tokens:
                    if t in ['+', '/']:
                        formatted_tokens.append(t)
                    elif t.strip():
                        formatted_tokens.append(_italicize_token(t))
                suffix = " isozyme" if is_isozyme else ""
                formatted_paren = f"(${''.join(formatted_tokens)}${suffix})"
                return f"{base_rxn} {formatted_paren}".strip() if base_rxn else formatted_paren

        is_gene_symbol = (
            bool(re.match(r'^[A-Z][a-z0-9]+(\*)?$', rxn_str)) or 
            bool(re.match(r'^[A-Z][a-z]+[0-9]*[a-zA-Z0-9]*(\*)?$', rxn_str))
        ) and not rxn_str.isupper() and '_' not in rxn_str
        
        if is_gene_symbol and len(rxn_str) <= 15:
            return f"${_italicize_token(rxn_str)}$"

        return rxn_str

    print("=== Initiating Module 2: Metabolic Characterization ===")
    os.makedirs(output_dir, exist_ok=True)
    sc.settings.figdir = output_dir
    plot_key = viz_method.lower()

    print(f"--> Loading Annotated RCS from {os.path.basename(adata_path)}...")
    adata = sc.read_h5ad(adata_path)
    
    if cluster_column not in adata.obs.columns:
        raise ValueError(f"CRITICAL ERROR: '{cluster_column}' not found in adata.obs. Available columns: {list(adata.obs.columns)}")

    if 'subsystem' not in adata.var.columns:
        print("WARNING: 'subsystem' column not found in adata.var. Defaulting to 'Unknown'.")
        adata.var['subsystem'] = 'Unknown'

    # -------------------------------------------------------------------------
    # 1) FEATURE PLOTS FOR SUBSTRING SEARCHED REACTIONS
    # -------------------------------------------------------------------------
    if target_reactions:
        print(f"--> Generating feature plots for {len(target_reactions)} target reaction patterns...")
        all_matching_rxns = set()
        for target_str in target_reactions:
            matching = [var for var in adata.var_names if target_str in var]
            if matching:
                all_matching_rxns.update(matching)
            else:
                print(f"WARNING: No features matched target substring '{target_str}'.")
                
        if all_matching_rxns:
            all_matching_rxns = list(all_matching_rxns)
            expr_data = adata[:, all_matching_rxns].X
            global_max = expr_data.max()
            
            magma_colors = plt.get_cmap('Purples')(np.linspace(0, 1, 256))
            magma_colors[0] = mcolors.to_rgba('lightgrey')
            custom_magma = mcolors.LinearSegmentedColormap.from_list('magma_grey_zero', magma_colors)
            
            for rxn in all_matching_rxns:
                safe_rxn_name = str(rxn).replace('/', '_').replace('\\', '_').replace(':', '_')
                ax = sc.pl.embedding(
                    adata, basis=plot_key, color=rxn, 
                    size=point_size, cmap=custom_magma, 
                    vmin=0, vmax=global_max, frameon=False, 
                    show=False
                )
                save_path = os.path.join(output_dir, f"{plot_key}_{safe_rxn_name}_capacity.png")
                ax.figure.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(ax.figure)

    # -------------------------------------------------------------------------
    # 2) DIFFERENTIAL ANALYSIS & RIGHT-ALIGNED BRACKETED BAR CHARTS
    # -------------------------------------------------------------------------
    print(f"--> Computing differential reactions and generating subsystem graphs for '{cluster_column}'...")
    sc.tl.rank_genes_groups(adata, groupby=cluster_column, method='wilcoxon')
    
    cluster_subsystem_dict = {}
    all_sig_cluster_rxns = pd.DataFrame()
    
    for cluster in adata.obs[cluster_column].cat.categories:
        df = sc.get.rank_genes_groups_df(adata, group=cluster)
        sig_df = df[(df['pvals_adj'] < pval_thresh) & (df['logfoldchanges'] > lfc_thresh)].copy()
        sig_df['cluster'] = cluster
        sig_df['subsystem'] = sig_df['names'].map(adata.var['subsystem']).fillna('Unknown')
        all_sig_cluster_rxns = pd.concat([all_sig_cluster_rxns, sig_df])
        
        top_20_sig = sig_df.head(20)
        subsystems = top_20_sig[top_20_sig['subsystem'] != 'Unknown']['subsystem'].unique().tolist()
        safe_cluster_name = str(cluster).replace('/', '_').replace('\\', '_')
        cluster_subsystem_dict[f"Cluster_{safe_cluster_name}"] = subsystems

        clean_sig_df = sig_df[sig_df['subsystem'] != 'Unknown'].copy()
        if not clean_sig_df.empty:
            # 1. Select top 15 reactions strictly by adjusted p-value
            top_15_rxns = clean_sig_df.sort_values(
                by=['pvals_adj', 'logfoldchanges'], 
                ascending=[True, False]
            ).head(15).copy()
            
            # 2. Group adjacently by subsystem
            sub_order = (
                top_15_rxns.groupby('subsystem', observed=True)['pvals_adj']
                .min()
                .sort_values(ascending=True)
                .index.tolist()
            )
            top_15_rxns['subsystem'] = pd.Categorical(top_15_rxns['subsystem'], categories=sub_order, ordered=True)
            top_15_rxns = top_15_rxns.sort_values(by=['subsystem', 'logfoldchanges'], ascending=[True, False]).reset_index(drop=True)
            
            # 3. Format reaction labels
            top_15_rxns['rich_label'] = top_15_rxns['names'].apply(_format_rich_label)
            plot_df = top_15_rxns.iloc[::-1].reset_index(drop=True)
            
            # Colormap scaled to -log10(padj)
            plot_df['neg_log10_padj'] = -np.log10(plot_df['pvals_adj'].clip(lower=1e-300))
            base_cmap = plt.get_cmap('Oranges')
            cmap = mcolors.LinearSegmentedColormap.from_list(
                'LighterOranges', base_cmap(np.linspace(0.25, 0.7, 256))
            )
            norm = plt.Normalize(vmin=plot_df['neg_log10_padj'].min() * 0.8, vmax=plot_df['neg_log10_padj'].max())
            bar_colors = [cmap(norm(val)) for val in plot_df['neg_log10_padj']]
            
            # Canvas geometry
            fig_w = 9.2
            fig_h = max(5.0, len(plot_df) * 0.42)
            fig = plt.figure(figsize=(fig_w, fig_h))
            
            # Fixed axis position and width
            ax_left = 0.28
            ax_w = 0.20
            ax_bottom = 0.12
            ax_h = 0.78
            ax = fig.add_axes([ax_left, ax_bottom, ax_w, ax_h])
            
            y_positions = np.arange(len(plot_df))
            ax.barh(y_positions, plot_df['logfoldchanges'], color=bar_colors, height=0.7, edgecolor='none')
            
            # Scale naturally to the reactions present (no artificial empty slots)
            ax.set_ylim(-0.5, len(plot_df) - 0.5)
            ax.set_yticks(y_positions)
            ax.set_yticklabels(plot_df['rich_label'], fontsize=9.5)
            ax.set_xlabel(r'$\log_2\ \mathrm{Fold\ Change}$', fontsize=10.5, fontweight='bold')
            ax.set_title(f'Top Enriched Reactions by Subsystem (Cluster {cluster})', fontsize=11.5, pad=15, fontweight='bold')
            
            # Independent x-axis maximum per cluster
            cluster_max_lfc = max(plot_df['logfoldchanges'].max(), 0.1)
            ax.set_xlim(left=0, right=cluster_max_lfc * 1.08)
            
            trans_blended = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
            
            bracket_x = 1.05
            bracket_tick_len = 0.03
            text_x = bracket_x + bracket_tick_len + 0.04
            
            for sub_name, group in plot_df.groupby('subsystem', observed=True):
                if group.empty:
                    continue
                y_start = group.index.min() - 0.32
                y_end = group.index.max() + 0.32
                y_mid = (y_start + y_end) / 2.0
                
                ax.plot(
                    [bracket_x, bracket_x + bracket_tick_len, bracket_x + bracket_tick_len, bracket_x],
                    [y_start, y_start, y_end, y_end],
                    transform=trans_blended,
                    color='black', lw=1.2, clip_on=False
                )
                
                ax.text(
                    text_x, y_mid, sub_name,
                    transform=trans_blended,
                    va='center', ha='left', fontsize=9.5, fontweight='bold',
                    color='#222222', clip_on=False
                )

            # >>> TUNE THIS X-COORDINATE (FIRST PARAMETER) TO SHIFT THE COLORBAR RIGHT <<<
            cbar_x_pos = 0.88  
            cax = fig.add_axes([cbar_x_pos, ax_bottom + 0.12, 0.022, ax_h * 0.70])
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_label(r'$-\log_{10}(p_{\mathrm{adj}})$', rotation=270, labelpad=18, fontsize=10.5, fontweight='bold')

            sns.despine(ax=ax, top=True, right=True)
            
            save_path = os.path.join(output_dir, f"subsystem_enrichment_Cluster_{safe_cluster_name}.png")
            fig.savefig(save_path, dpi=300)
            plt.close(fig)

    # Save cluster spreadsheets & summary JSON
    all_sig_cluster_rxns.to_csv(os.path.join(output_dir, "filtered_enriched_reactions_by_cluster.csv"), index=False)
    with open(os.path.join(output_dir, "top20_subsystems_by_cluster.json"), 'w') as f:
        json.dump(cluster_subsystem_dict, f, indent=4)

    # -------------------------------------------------------------------------
    # 3) DOT PLOTS (CLUSTERS & CELL TYPES)
    # -------------------------------------------------------------------------
    print("--> Generating dot plots for cluster-enriched reactions...")
    dp_cluster = sc.pl.rank_genes_groups_dotplot(
        adata, n_genes=5, min_logfoldchange=lfc_thresh, 
        show=False, return_fig=True
    )
    dp_cluster.savefig(os.path.join(output_dir, "cluster_enriched_reactions.png"))

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
    dp_celltype = sc.pl.rank_genes_groups_dotplot(
        adata, n_genes=5, key='rank_genes_celltype', 
        min_logfoldchange=lfc_thresh, show=False, return_fig=True
    )
    dp_celltype.savefig(os.path.join(output_dir, "celltype_enriched_reactions.png"))

    print(f"=== Module 2 Complete. All characterization outputs saved to: {output_dir} ===")
    return adata