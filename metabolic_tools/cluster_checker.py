import os
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.metrics import silhouette_score

def resolution_sweep(adata_path, resolutions, output_dir="results", n_top_reactions=2000):
    """
    Loads an RCS matrix, identifies HVRs, computes PCA/Neighbors,
    performs a Leiden resolution sweep, and saves the plots and processed object.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading matrix from {adata_path}...")
    adata = sc.read_h5ad(adata_path)
    
    print("Identifying highly variable reactions and computing PCA/Neighbors...")
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_reactions)
    sc.tl.pca(adata, use_highly_variable=True)
    sc.pp.neighbors(adata)

    if isinstance(resolutions, tuple) and len(resolutions) == 3:
        res_array = np.arange(resolutions[0], resolutions[1], resolutions[2])
    else:
        res_array = resolutions

    sweep_results = []
    X_pca = adata.obsm['X_pca']
    
    # Isolate HVRs once before the loop (for fast matrix extraction)
    adata_hvr = adata[:, adata.var['highly_variable']]

    print("Beginning Leiden resolution sweep...")
    for res in res_array:
        # Format key to prevent float errors (e.g., leiden_0.10)
        cluster_key = f'leiden_{res:.2f}'
        
        sc.tl.leiden(adata, resolution=res, key_added=cluster_key, flavor='igraph', n_iterations=2, directed=False)
        
        cluster_entropies = []
        
        # EXTRACT LABELS FROM THE MAIN ADATA OBJECT (Which has the new column)
        labels = adata.obs[cluster_key]
        
        for cluster in labels.unique():
            cells = labels == cluster
            scores = adata_hvr.X[cells] # Slice the HVR matrix using the main labels
            
            if hasattr(scores, "toarray"):
                scores = scores.toarray()
                
            reaction_entropies = [entropy(np.histogram(scores[:, i], bins=30)[0]) for i in range(scores.shape[1])]
            cluster_entropies.append(np.mean(reaction_entropies))
            
        mean_e = np.mean(cluster_entropies)
        sil_score = silhouette_score(X_pca, labels) if len(np.unique(labels)) > 1 else np.nan
            
        sweep_results.append({'Resolution': res, 'Mean_Entropy': mean_e, 'Silhouette_Score': sil_score})
        print(f"Resolution {res:.2f} | Mean Entropy: {mean_e:.3f} | Silhouette Score: {sil_score:.3f}")

    # Visualize and Save
    results_df = pd.DataFrame(sweep_results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(results_df['Resolution'], results_df['Mean_Entropy'], marker='o', color='tab:blue', linewidth=2)
    axes[0].set_title('Metabolic Purity (HVR Shannon Entropy)')
    axes[0].set_xlabel('Leiden Resolution (γ)')
    axes[0].set_ylabel('Mean Cluster Entropy (Lower = Purer)')
    axes[0].grid(alpha=0.3)

    axes[1].plot(results_df['Resolution'], results_df['Silhouette_Score'], marker='s', color='tab:green', linewidth=2)
    axes[1].set_title('Cluster Separation (PCA Silhouette Score)')
    axes[1].set_xlabel('Leiden Resolution (γ)')
    axes[1].set_ylabel('Silhouette Score (Higher = Better)')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    image_output_path = os.path.join(output_dir, "sweep_results.png")
    plt.savefig(image_output_path, dpi=300, bbox_inches='tight') 
    plt.close()
    
    data_output_path = os.path.join(output_dir, "reaction_scores_hvr_processed.h5ad")
    print(f"Saving pre-processed object to {data_output_path}...")
    adata.write(data_output_path)
    
    return adata