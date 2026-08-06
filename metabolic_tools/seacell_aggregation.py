import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
import SEACells
import os
import matplotlib.pyplot as plt
import logging
import warnings

def run_stratified_metacells(
    input_h5ad_path,              # str: Path to your cleaned, post-QC single-cell .h5ad file.
    celltype_column,              # str: The exact column name in adata.obs holding your cell annotations.
    output_h5ad_path=None,        # str (optional): Where to save the final dense metacell .h5ad file.
    raw_layer_name=None,          # str (optional): The adata.layer with raw counts (e.g., 'counts'). If None, mathematically reverses log1p.
    target_metacell_size=50,      # int: The ideal number of single cells to merge into one metacell.
    n_components=50,              # int: Number of Principal Components (PCs) to compute for the manifold.
    n_top_genes=1500,             # int: Number of Highly Variable Genes (HVGs) to select before PCA.
    min_iter=10,                  # int: Minimum number of training steps for the SEACell algorithm.
    max_iter=150,                 # int: Maximum number of training steps (prevents infinite loops).
    convergence_epsilon=1e-4,     # float: The mathematical threshold to decide when archetypes have settled.
    visualization_method='tsne',  # str: Choose 'tsne', 'umap', or None for the final visual check.
    plot_save_path=None,          # str (optional): Where to save the plot image (e.g., 'my_umap.png').
    tsne_perplexity=30,           # int/float: Tuning for t-SNE. Higher values consider more global structure.
    umap_min_dist=0.1,            # float: Tuning for UMAP. Lower values (e.g., 0.05) force tighter, island-like clusters.
    umap_n_neighbors=15           # int: Tuning for UMAP. Number of neighboring points used to build the graph.
):
    """
    Computes Stratified Metacells across distinct cell types to prevent chimeras,
    preserves mass balance for metabolic modeling, and generates a visual overlay.
    """
    
    # =========================================================
    # 0. DYNAMIC ENVIRONMENT PATCH
    # Safely patches AnnData to prevent SEACells from crashing 
    # due to the deprecated 'dtype' argument in modern AnnData versions.
    # =========================================================
    if not hasattr(ad.AnnData, '_is_patched_for_seacells'):
        _original_init = ad.AnnData.__init__
        
        def _patched_init(self, *args, **kwargs):
            kwargs.pop('dtype', None) # Remove the illegal argument
            _original_init(self, *args, **kwargs)
            
        ad.AnnData.__init__ = _patched_init
        ad.AnnData._is_patched_for_seacells = True # Mark as patched to prevent recursion
    
    # =========================================================
    # 1. DATA LOADING & MASS BALANCE PREP
    # =========================================================
    print(f"Loading dataset from {input_h5ad_path}...")
    if not os.path.exists(input_h5ad_path):
        raise FileNotFoundError(f"Could not find {input_h5ad_path}")
        
    adata = sc.read_h5ad(input_h5ad_path)
    
    if celltype_column not in adata.obs.columns:
        raise ValueError(f"Column '{celltype_column}' not found in adata.obs")
    
    # Mass Balance Fail-Safe
    if raw_layer_name is None:
        print("   WARNING: No raw layer specified. Reversing log1p (e^x - 1) to restore linear stoichiometry...")
        adata.layers['restored_linear_counts'] = np.expm1(adata.X)
        summarization_layer = 'restored_linear_counts'
    else:
        summarization_layer = raw_layer_name

    # =========================================================
    # 2. GLOBAL GEOMETRY (HVGs, PCA, Manifold)
    # =========================================================
    print("Computing Global Manifold (HVGs, PCA)...")
    if 'highly_variable' not in adata.var:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat')
    if 'X_pca' not in adata.obsm:
        sc.tl.pca(adata, n_comps=n_components, use_highly_variable=True)
        
    # --- VISUALIZATION COMPUTATION ---
    if visualization_method == 'umap' and 'X_umap' not in adata.obsm:
        print("   Computing UMAP coordinates...")
        if 'neighbors' not in adata.uns:
            sc.pp.neighbors(adata, n_pcs=n_components, n_neighbors=umap_n_neighbors)
        sc.tl.umap(adata, min_dist=umap_min_dist)
        
    elif visualization_method == 'tsne' and 'X_tsne' not in adata.obsm:
        print("   Computing t-SNE coordinates...")
        sc.tl.tsne(adata, n_pcs=n_components, perplexity=tsne_perplexity)

    adata.obs['Stratified_SEACell'] = 'unassigned'

    # =========================================================
    # 3. STRATIFIED SEACELL LOOP
    # =========================================================
    print(f"Beginning Stratified Aggregation across '{celltype_column}'...")
    cell_types = adata.obs[celltype_column].unique()
    
    # Suppress verbose loop outputs and minor convergence bounces
    logging.getLogger().setLevel(logging.ERROR) 
    warnings.filterwarnings('ignore', message='.*Algorithm has not converged.*')
    
    for ct in cell_types:
        subset = adata[adata.obs[celltype_column] == ct].copy()
        n_cells = subset.n_obs
        
        # Skip dead clusters
        if n_cells < 3: 
            print(f"   - Skipping {ct}: Only {n_cells} cells.")
            continue
            
        n_SEACells = int(np.round(n_cells / target_metacell_size))
        
        # Dynamic Bypass: If math dictates 1 or fewer metacells, skip the ML algorithm
        if n_SEACells <= 1:
            print(f"   - {ct}: {n_cells} cells -> Bypassing algorithm, grouped into 1 macro-cell.")
            adata.obs.loc[subset.obs_names, 'Stratified_SEACell'] = f"{ct}_Metacell_0"
            continue 
        
        print(f"   - {ct}: {n_cells} cells -> Grouping into {n_SEACells} metacells.")

        model = SEACells.core.SEACells(
            subset, 
            build_kernel_on='X_pca', 
            n_SEACells=n_SEACells, 
            n_waypoint_eigs=min(10, n_cells - 1), 
            convergence_epsilon=convergence_epsilon
        )
        
        try:
            model.construct_kernel_matrix()
            model.initialize_archetypes()
            model.fit(min_iter=min_iter, max_iter=max_iter)
            
            # Map subset labels back to the master object with a unique cell type prefix
            unique_labels = ct + "_Metacell_" + subset.obs['SEACell'].astype(str)
            adata.obs.loc[subset.obs_names, 'Stratified_SEACell'] = unique_labels
            
        except Exception as e:
            # Failsafe: Preserve data by forcing a macro-cell if a cluster's geometry crashes
            print(f"     ! Error in {ct} ({e}). Forcing into 1 macro-cell to preserve data.")
            adata.obs.loc[subset.obs_names, 'Stratified_SEACell'] = f"{ct}_Metacell_0"

    # Restore warnings and logging
    logging.getLogger().setLevel(logging.INFO) 
    warnings.resetwarnings()

    # Drop unassigned stray cells
    adata = adata[adata.obs['Stratified_SEACell'] != 'unassigned'].copy()
    
    # Overwrite the default column SEACells expects to look for
    adata.obs['SEACell'] = adata.obs['Stratified_SEACell']

    # =========================================================
    # 4. VISUALIZATION CHECK
    # =========================================================
    if visualization_method in ['tsne', 'umap']:
        print(f"Generating Global {visualization_method.upper()} overlay of Stratified SEACells...")
        
        plt.figure(figsize=(8, 8))
        ax = plt.gca() 
        
        # Determine the correct scanpy plot basis and SEACell key dynamically
        sc_basis = 'umap' if visualization_method == 'umap' else 'tsne'
        seacell_key = 'X_umap' if visualization_method == 'umap' else 'X_tsne'
        
        # Plot background scatter
        sc.pl.scatter(adata, basis=sc_basis, color=celltype_column, frameon=False, show=False, ax=ax, alpha=0.3)
        
        # Overlay archetypes naturally on the active canvas
        SEACells.plot.plot_2D(adata, key=seacell_key, colour_metacells=True)
        
        if plot_save_path:
            plt.savefig(plot_save_path, bbox_inches='tight', dpi=300)
            print(f"   Plot saved to {plot_save_path}")
        plt.show()

    # =========================================================
    # 5. GLOBAL SUMMARIZATION & METADATA
    # =========================================================
    print(f"Aggregating into Dense Metacells using layer: '{summarization_layer}'...")
    final_metacell_adata = SEACells.core.summarize_by_SEACell(
        adata, 
        SEACells_label='SEACell', 
        summarize_layer=summarization_layer
    )
    
    # Stamp guaranteed purity metrics
    final_metacell_adata.obs['majority_celltype'] = [label.split('_Metacell_')[0] for label in final_metacell_adata.obs_names]
    final_metacell_adata.obs['celltype_purity'] = 1.0

    # Clean up massive unlogged arrays from memory
    if 'restored_linear_counts' in adata.layers:
        del adata.layers['restored_linear_counts']
        
    print(f"Aggregation complete. Final matrix dimensions: {final_metacell_adata.shape}")
    
    # =========================================================
    # 6. OUTPUT
    # =========================================================
    if output_h5ad_path:
        print(f"Saving to {output_h5ad_path}...")
        final_metacell_adata.write_h5ad(output_h5ad_path)
        print("Save successful!")
        
    return final_metacell_adata