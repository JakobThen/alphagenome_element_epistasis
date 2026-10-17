"""Compute AlphaGenome ISM attributions for a set of genomic regions.

For every region in a BED file, scores all single-nucleotide substitutions with
AlphaGenome inside a larger sequence context and collapses them into per-base
attribution matrices of shape (region length, 4). Accessibility (ATAC) is
always computed; predicted TF ChIP-seq and RNA expression of named target genes
are added when ``--tfs`` and ``--genes`` are given. Results are written to one
HDF5 file, ``<out_prefix>_<bed name>.h5``, readable with
``element_epistasis.ism.load_ism_dict``.

Example:
    python src/ism_attribution/run_ism.py \\
        --bed data/raw/enhancers.bed --genes data/raw/target_genes.csv \\
        --tfs data/raw/tfs.tsv --curies EFO:0002067 --seq_len 1MB \\
        --out_prefix data/processed/K562_ISM_ --api_key "$ALPHAGENOME_API_KEY"
"""

import os
import sys
import argparse
import pandas as pd
from joblib import Parallel, delayed

from alphagenome import colab_utils
from alphagenome.data import gene_annotation, genome, transcript as transcript_utils
from alphagenome.interpretation import ism
from alphagenome.models import dna_client, variant_scorers
from alphagenome.visualization import plot_components

# Make the package in ../ importable when the script is run without installing it.
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from element_epistasis.ism import collect_ism_scores, build_ism_matrix, save_ism_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Run AlphaGenome ISM pipeline for targeted regions.")
    parser.add_argument("--bed", required=True, type=str, help="Path to BED file. If 4th column is missing, regions are numbered sequentially.")
    parser.add_argument("--curies", required=True, nargs='+', type=str,  help="Space-separated list of track curies (e.g., EFO:0002067).")
    parser.add_argument("--seq_len", required=True, type=str, choices=["16KB", "100KB", "500KB", "1MB"], help="Sequence length context for the model.")
    parser.add_argument("--out_prefix", required=True, type=str, help="Prefix for the output H5 file.")
    parser.add_argument("--tfs", nargs='+', type=str, default=None, help="Space-separated list of TF names, OR path to a single .txt file with one TF per line.")
    parser.add_argument("--genes", type=str, default=None,help="Path to CSV file with target genes. Rows match BED intervals, multiple columns allowed, no header.")
    parser.add_argument("--api_key", type=str, required=True,help="AlphaGenome API key.")
    return parser.parse_args()


def get_seq_length(seq_str: str):
    """Maps CLI string to dna_client sequence length constant."""
    mapping = {
        "16KB": dna_client.SEQUENCE_LENGTH_16KB,
        "100KB": dna_client.SEQUENCE_LENGTH_100KB,
        "500KB": dna_client.SEQUENCE_LENGTH_500KB,
        "1MB": dna_client.SEQUENCE_LENGTH_1MB,
    }
    return mapping[seq_str]


def process_region_dynamic(name, ism_data, curie, indices, genes=None, TFs=None):
    """
    Modified process_region that accepts a dynamic dictionary mapping 
    modalities to their list indices (e.g. {"ATAC": 0, "TF": 1}).
    """
    tmp = {}
    
    # ATAC is always present and always at index 0
    atac_col = collect_ism_scores(ism_data, curie=curie, modality={"ATAC": indices["ATAC"]})
    tmp["ATAC"] = build_ism_matrix(atac_col)
    
    if genes and "RNA" in indices:
        rna_col = collect_ism_scores(ism_data, curie=curie, modality={"RNA": indices["RNA"]})
        if rna_col is not None:
            for gene in genes:
                tmp[f"RNA {gene}"] = build_ism_matrix(rna_col, target=gene)

    if TFs and "TF" in indices:
        tf_col = collect_ism_scores(ism_data, curie=curie, modality={"TF": indices["TF"]})
        if tf_col is not None:
            for tf in TFs:
                tmp[f"TF {tf}"] = build_ism_matrix(tf_col, target=tf)

    return name, tmp


def main():
    args = parse_args()
    
    if not args.api_key:
        sys.exit("Error: AlphaGenome API key is required. Pass via --api_key or set ALPHAGENOME_API_KEY.")
    
    # Initialize Model
    model = dna_client.create(args.api_key)
    
    # Setup Output File
    bed_basename = os.path.basename(args.bed).replace(".bed", "")
    out_file = f"{args.out_prefix}_{bed_basename}.h5"
    print(f"Output will be saved to: {out_file}")

    # Parse BED File
    bed_raw = pd.read_csv(args.bed, sep="\t", header=None)
    if bed_raw.shape[1] >= 4:
        enhancers = bed_raw.iloc[:, :4]
        enhancers.columns = ["chrom", "start", "end", "name"]
    else:
        enhancers = bed_raw.iloc[:, :3]
        enhancers.columns = ["chrom", "start", "end"]
        enhancers["name"] = [f"region_{i}" for i in range(len(enhancers))]
        
    print(f"Loaded {len(enhancers)} regions from {args.bed}")

    # Parse Genes CSV (if provided)
    if args.genes:
        genes_df = pd.read_csv(args.genes, header=None)
        if len(genes_df) != len(enhancers):
            sys.exit(f"Error: Target genes file has {len(genes_df)} rows, but BED file has {len(enhancers)} rows.")
        # Collect non-NaN values across columns into a list per row
        enhancers["target_gene"] = [
            row.dropna().astype(str).tolist() for _, row in genes_df.iterrows()
        ]
    else:
        enhancers["target_gene"] = None

    # Parse TFs (if provided)
    tfs = []
    if args.tfs:
        if len(args.tfs) == 1 and os.path.isfile(args.tfs[0]):
            with open(args.tfs[0], 'r') as f:
                tfs = [line.strip() for line in f if line.strip()]
        else:
            tfs = args.tfs
            
    # Filter TFs against available metadata
    track_metadata = model.output_metadata(dna_client.Organism.HOMO_SAPIENS).concatenate()
    metadata = track_metadata[
        (track_metadata["ontology_curie"].isin(args.curies)) & 
        (track_metadata["output_type"].isin([
            dna_client.OutputType.ATAC, 
            dna_client.OutputType.CHIP_TF, 
            dna_client.OutputType.RNA_SEQ
        ]))
    ].dropna(how="all", axis=1)

    if tfs:
        valid_tfs = metadata.get("transcription_factor", pd.Series(dtype=str)).str.upper().tolist()
        tfs = [tf.upper() for tf in tfs if tf.upper() in valid_tfs]
        print(f"Filtered to {len(tfs)} valid transcription factors for given curies.")
        if len(tfs) == 0: tfs = None

    # Dynamically build Scorers and Modality Indices
    scorers = [variant_scorers.RECOMMENDED_VARIANT_SCORERS['ATAC']]
    modality_indices = {"ATAC": 0}
    current_idx = 1
    
    if tfs:
        scorers.append(variant_scorers.RECOMMENDED_VARIANT_SCORERS['CHIP_TF'])
        modality_indices["TF"] = current_idx
        current_idx += 1
        
    if args.genes:
        scorers.append(variant_scorers.RECOMMENDED_VARIANT_SCORERS['RNA_SEQ'])
        modality_indices["RNA"] = current_idx

    # Generate model intervals
    seq_len = get_seq_length(args.seq_len)
    enhancers["ivs"] = [
        genome.Interval(chromosome=row.chrom, start=row.start, end=row.end) 
        for row in enhancers.itertuples()
    ]
    enhancers["ivs_big"] = [iv.resize(seq_len) for iv in enhancers["ivs"]]

    # Run AlphaGenome ISM
    print("Running Model ISM...")
    ism_result = {
        row.name: model.score_ism_variants(
            interval=row.ivs_big,
            ism_interval=row.ivs,
            variant_scorers=scorers,
            interval_variant=None,

        ) for row in enhancers.itertuples()
    }

    # Process results in parallel
    print("Processing ISM scores...")
    results = Parallel(n_jobs=-3, backend="loky", verbose=10)(
        delayed(process_region_dynamic)(
            name=row.name,
            ism_data=ism_result[row.name],
            curie=args.curies,
            indices=modality_indices,
            genes=row.target_gene,
            TFs=tfs
        )
        for row in enhancers.itertuples()
    )

    # Save output
    print("Saving HDF5 file...")
    ism_enhancer = dict(results)
    save_ism_dict(ism_enhancer, out_file)
    print("Done!")


if __name__ == "__main__":
    main()
