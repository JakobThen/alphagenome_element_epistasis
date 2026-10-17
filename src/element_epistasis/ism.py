"""Collect AlphaGenome ISM scores into per-base attribution matrices and store them as HDF5."""

import h5py
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from typing import Any

from alphagenome import colab_utils
from alphagenome.data import gene_annotation, genome
from alphagenome.data import transcript as transcript_utils
from alphagenome.interpretation import ism
from alphagenome.models import dna_client, variant_scorers
from alphagenome.visualization import plot_components



def aggregate_adata_vars(
    adata: ad.AnnData, 
    groupby_cols: list[str] = ['ontology_curie', 'transcription_factor']
) -> ad.AnnData:
    """Average AnnData tracks that share the same values in the grouping columns.

    Args:
        adata: Variant scores, one variable per output track.
        groupby_cols: Track metadata columns to group on.

    Returns:
        AnnData with one variable per group, holding the mean score.
    """
    grouped = adata.var.groupby(groupby_cols)
    new_X_cols = []
    new_var_rows = []
    
    for name, group in grouped:
        X_sub = adata[:, group.index].X
        
        # Handle sparse vs dense matrix differences when computing the mean
        if sparse.issparse(X_sub):
            col_mean = np.asarray(X_sub.mean(axis=1))
        else:
            col_mean = np.asarray(X_sub.mean(axis=1, keepdims=True))
            
        new_X_cols.append(col_mean)
        
        var_repr = group.iloc[0].copy()
        # Flatten tuple names into a single string for the new index
        var_repr.name = "_".join([str(n) for n in name])
        new_var_rows.append(var_repr)
        
    new_adata = ad.AnnData(
        X=np.hstack(new_X_cols), 
        obs=adata.obs.copy(), 
        var=pd.DataFrame(new_var_rows)
    )
    
    # Carry over unstructured data and embeddings
    new_adata.uns = adata.uns.copy()
    if hasattr(adata, 'obsm'):
        new_adata.obsm = adata.obsm.copy()
        
    return new_adata


def average_all_stranded_expression(adata: ad.AnnData) -> ad.AnnData:
    """Average all RNA tracks into one, using only tracks on the gene's strand or unstranded.

    Args:
        adata: RNA variant scores, one observation per gene.

    Returns:
        AnnData with a single variable holding the strand-aware mean per gene.
    """
    gene_strands = adata.obs['strand'].astype(str).values
    track_strands = adata.var['strand'].astype(str).values
    
    # Mask is True if strands match or if the track is unstranded ('.')
    mask = (gene_strands[:, None] == track_strands[None, :]) | (track_strands[None, :] == '.')
    
    # Sum up valid expression values
    if sparse.issparse(adata.X):
        valid_sum = np.asarray(adata.X.multiply(mask).sum(axis=1))
    else:
        valid_sum = np.sum(np.where(mask, adata.X, 0.0), axis=1, keepdims=True)
        
    valid_count = mask.sum(axis=1, keepdims=True)
    
    # Safely compute the mean, replacing division-by-zero NaNs with 0.0
    with np.errstate(divide='ignore', invalid='ignore'):
        col_mean = valid_sum / valid_count
    col_mean = np.nan_to_num(col_mean, nan=0.0)
    
    # Create the unified variable metadata
    new_var = pd.DataFrame([{
        'name': 'mean_expression_all_tracks',
        'strand': '.', 
        'Assay title': 'Aggregated RNA-seq'
    }], index=['mean_expression_all_tracks'])
    
    new_adata = ad.AnnData(
        X=col_mean, 
        obs=adata.obs.copy(), 
        var=new_var
    )
    
    new_adata.uns = adata.uns.copy()
    if hasattr(adata, 'obsm'):
        new_adata.obsm = adata.obsm.copy()
        
    return new_adata
   

def extract_var_score(adata: ad.AnnData, curie: list[str]) -> tuple[list[str], np.ndarray] | None:
    """Extract target names and scores of one variant for the given curies.

    Args:
        adata: Scores of one variant for one output type.
        curie: Ontology curies to keep, e.g. ["EFO:0002067"].

    Returns:
        Tuple of (target names, scores): "ATAC" for accessibility, factor names
        for TF ChIP-seq and gene names for RNA-seq. None for unsupported assays.
    """
    assay_type = adata.var["Assay title"].iloc[0]
    adata = adata[:, (adata.var.ontology_curie.isin(curie))]
    
    if assay_type == "ATAC-seq":
        adata = aggregate_adata_vars(adata, groupby_cols=['strand'])
        return (["ATAC"], adata.X.ravel())
        
    elif assay_type == "TF ChIP-seq":
        adata = aggregate_adata_vars(adata, groupby_cols=['transcription_factor'])
        return (list(adata.var['transcription_factor']), adata.X.ravel())
        
    elif assay_type in ['polyA plus RNA-seq', 'total RNA-seq']:
        # Return empty arrays if no genes fall in the window
        if adata.obs.empty: 
            return ([], np.array([]))
            
        adata = aggregate_adata_vars(adata, groupby_cols=['strand'])
        adata = average_all_stranded_expression(adata)
        return (list(adata.obs["gene_name"]), adata.X.ravel())
        
    else:
        print(f"WARNING: Modality '{assay_type}' not handled yet.")
        return None
    

def collect_ism_scores(
    ism_result: list[list[ad.AnnData]], 
    curie: list[str], 
    modality: dict[str, int]
) -> dict[str, Any] | None:
    """Gather the scores of every ISM variant for one output type into a matrix.

    Args:
        ism_result: AlphaGenome ISM output, one list of AnnData per variant.
        curie: Ontology curies to keep.
        modality: Single-entry mapping of output name to its index in each
            variant's list, e.g. {"ATAC": 0}.

    Returns:
        Dictionary with "targets", "variants" and a (variants x targets) "scores"
        matrix, or None if no target is present.
    """
    m_idx = list(modality.values())[0]

    # Establish output shape based on the first variant
    first_pass = extract_var_score(ism_result[0][m_idx], curie=curie)
    if not first_pass or len(first_pass[0]) == 0:
        return None
        
    first_targets, first_scores = first_pass

    n_variants = len(ism_result)
    n_targets = len(first_targets)
    
    score_matrix = np.full((n_variants, n_targets), np.nan)
    variants = []

    # Initialize with the first variant we already computed
    score_matrix[0, :] = first_scores
    variants.append(ism_result[0][m_idx].uns["variant"])

    # Process remaining variants
    for i, adata_list in enumerate(ism_result[1:], start=1):
        _, scores = extract_var_score(adata_list[m_idx], curie=curie)
        score_matrix[i, :] = scores
        variants.append(adata_list[m_idx].uns["variant"])

    return {
        "targets": first_targets,
        "variants": variants,
        "scores": score_matrix,
    }


def build_ism_matrix(collected: dict[str, Any] | None, target: str | None = None) -> Any:
    """Turn collected scores for one target into a per-base attribution matrix.

    Args:
        collected: Output of collect_ism_scores.
        target: Target name for TF or RNA outputs; None uses the only column (ATAC).

    Returns:
        Array of shape (region length, 4) with scores at the reference base, or
        None if the target is missing.
    """
    if collected is None:
        return None

    if target is None:
        col_idx = 0
    else:
        target_upper = target.upper()
        targets_upper = [t.upper() for t in collected["targets"]]
        
        if target_upper not in targets_upper:
            return None
            
        col_idx = targets_upper.index(target_upper)

    v_scores = list(collected["scores"][:, col_idx])

    return ism.ism_matrix(
        variant_scores=v_scores,
        variants=collected["variants"],
        multiply_by_sequence=True,
        require_fully_filled=False,
    )


def process_region(
    name: str, 
    ism_data: list[list[ad.AnnData]], 
    curie: list[str], 
    genes: list[str] | None = None, 
    TFs: list[str] | None = None
) -> tuple[str, dict[str, Any]]:
    """Build the ATAC, RNA and TF attribution matrices of one region.

    Args:
        name: Region name.
        ism_data: AlphaGenome ISM output for the region.
        curie: Ontology curies to keep.
        genes: Target genes for RNA attributions; skipped when None.
        TFs: Factors for TF ChIP-seq attributions; skipped when None.

    Returns:
        Tuple of (name, mapping of track label to attribution matrix).
    """
    tmp = {}
    
    atac_col = collect_ism_scores(ism_data, curie=curie, modality={"ATAC": 0})
    tmp["ATAC"] = build_ism_matrix(atac_col)
    
    if genes:
        rna_col = collect_ism_scores(ism_data, curie=curie, modality={"RNA": 2})
        if rna_col is not None: 
            for gene in genes:
                tmp[f"RNA {gene}"] = build_ism_matrix(rna_col, target=gene)

    if TFs:
        tf_col = collect_ism_scores(ism_data, curie=curie, modality={"TF": 1})
        if tf_col is not None:
            for tf in TFs:
                tmp[f"TF {tf}"] = build_ism_matrix(tf_col, target=tf)

    return name, tmp


def save_ism_dict(ism_enhancer: dict[str, dict[str, Any]], path: str) -> None:
    """Write attribution matrices to HDF5, one group per region and one dataset per track.

    Track labels are stored with spaces replaced by underscores; missing matrices
    are skipped.

    Args:
        ism_enhancer: Mapping of region name to track label to matrix.
        path: Output HDF5 file.
    """
    with h5py.File(path, "w") as f:
        for enhancer_name, modalities in ism_enhancer.items():
            grp = f.create_group(enhancer_name)
            
            for modality_name, matrix in modalities.items():
                if matrix is None:
                    continue
                    
                # HDF5 keys don't support spaces
                key = modality_name.replace(" ", "_")
                grp.create_dataset(key, data=matrix, compression="gzip", compression_opts=4)


def load_ism_dict(
    path: str, 
    enhancers: list[str] | None = None, 
    modalities: list[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Load attribution matrices written by save_ism_dict.

    Regions are returned in the file's alphabetical order, which defines seq_idx
    in the notebooks. Track labels get their space back ("RNA_GENE" -> "RNA GENE").

    Args:
        path: HDF5 file.
        enhancers: Regions to load; all when None.
        modalities: Track name prefixes to load, e.g. ["ATAC"]; all when None.

    Returns:
        Mapping of region name to track label to array of shape (length, 4).
    """
    result = {}
    
    with h5py.File(path, "r") as f:
        enhancer_keys = enhancers if enhancers is not None else list(f.keys())
        
        for enhancer_name in enhancer_keys:
            if enhancer_name not in f:
                continue
                
            grp = f[enhancer_name]
            result[enhancer_name] = {}
            
            for key in grp.keys():
                if modalities is not None and not any(key.startswith(m) for m in modalities):
                    continue
                    
                # Restore the original spacing convention (e.g. "RNA_GENE" -> "RNA GENE")
                modality_name = key.replace("_", " ", 1)
                result[enhancer_name][modality_name] = grp[key][:]
                
    return result