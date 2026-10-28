# Data formats

This folder holds the inputs the analysis needs and the tables it produces. To run the analysis on your own elements, replace the files in `raw/` (and, if you like, `meta/`) with files in the formats below, then run the notebooks. All locations are defined in `element_epistasis.paths`, so files must keep the names given here or the names must be changed there.

```
data/
├── raw/         your elements and measured CRISPRi coefficients
├── meta/        motif database
├── processed/   ISM attributions, AlphaGenome predictions, derived tables
└── reference/   hg38 FASTA and annotation cache (not version controlled)
```

## Naming convention

Every element is named `GENE_LABEL`, for example `ALDH1A1_enh2`.

- `GENE` is the target gene symbol exactly as it appears in GENCODE v46 (`gene_name`), and the gene must be protein coding. Everything before the first underscore is read as the gene, so the symbol itself must not contain an underscore.
- `LABEL` identifies the element within that gene and may contain anything but an underscore or a colon (`enh1`, `enh2-3`).
- An element regulating several genes is listed once, with its names joined by a colon: `ALDH1A1_enh2:ANXA1_enh1`. It is then analysed separately for each gene, inside that gene's window.

The same names must be used in every file below, because the tables are joined on them.

## `raw/`: elements and measured coefficients

`raw/` ships with small dummy files (3 genes with 3 elements each, invented coefficients, approximate coordinates that are not real regulatory elements) in every format described below. Replace them with your own elements and measurements. `processed/` starts empty and is filled by the pipeline.

### `enhancers.bed` (required)

Tab-separated, no header, one row per element, 0-based half-open coordinates on hg38. All elements must have the same length, because the dinucleotide shuffles are stacked into one tensor. The motif notebook assumes 313 bp (`ENH_LEN`); change that constant if yours differ. Each gene's elements and gene body must fit inside one 1 Mb prediction window, which notebook 01 asserts.

### `target_genes.csv` (required for the ISM job)

Comma-separated, no header, one row per BED row and in the same order. Each row lists the gene(s) whose predicted RNA attribution should be stored for that element; leave trailing fields empty when there are fewer genes.

```
ALDH1A1,
ALDH1A1,ANXA1
ANXA1,
```

### `tfs.tsv` (required)

One TF symbol per line, no header. Case and duplicates do not matter. The list is a prior: when a seqlet matches several motifs, a match to a listed factor is preferred as its label, and in the cross-enhancer design seqlets matching a listed factor are selected first. A family stem such as `RUNX` matches any motif name containing it.

### `enhancer_betas.csv` (required for validation)

Measured single-element effects. Comma-separated with a header.

| Column | Content |
|---|---|
| `enhancer` | element name (`GENE_LABEL`) |
| `beta` | measured effect of perturbing the element on the target gene |
| `source_dataset` | screen or dataset identifier |

Repeated rows for one element are averaged.

### `pair_interaction_adj.csv` (required for validation)

Measured pairwise interaction coefficients. Comma-separated with a header.

| Column | Content |
|---|---|
| `enhA`, `enhB` | element names (`GENE_LABEL`) of the two elements, both from the same gene |
| `interaction_beta` | measured interaction coefficient |
| `source_file` | screen or dataset identifier |

Repeated pairs are averaged. **List each pair in BED order** (`enhA` above `enhB` in `enhancers.bed`): the enhancer-level notebook joins on the two columns as written, and a pair in the opposite order is silently left out of the comparison. The motif notebook sorts pairs itself.

### `enhancer_interactions_exp.csv` (optional)

The same measurements in a second layout, with one `pair_name` column(`GENE_enhA+enhB`) plus `interaction_term`, `p_value` and `dataset`. Notebook 01 loads it for inspection only; `element_epistasis.cross_level.normalise_measured_pairs` accepts either layout.

### Units and sign of the measured coefficients

The model-side coefficients are natural-log fold changes against the unperturbed sequence: removing an activating element gives a negative main effect, and an interaction below zero is super-additive (the pair loses more than the singles predict). Provide measured coefficients on the same scale and with the same sign convention. Sign concordance and correlations do not depend on the log base, but the calibration slope does.

## `meta/`: motif database

`JASPAR2026_CORE_vertebrates_non-redundant_pfms_meme.txt`, but any motif
collection in MEME format works. The factor name is read as the second token of each `MOTIF` line, so the file must name motifs as JASPAR does:

```
MOTIF MA0035.5 GATA1
MOTIF MA0140.3 GATA1::TAL1
```

Heterodimer partners joined with `::` are resolved against the TF list. Point `paths.JASPAR_MEME` at a different file to use another collection.

## `processed/`: produced by the pipeline

These files are written by the ISM job and the notebooks. They are documented so that predictions from another source can be dropped in.

### `K562_ISM__enhancers.h5`, ISM attributions

Written by `src/ism_attribution/compute_enhancer_ISM.sh`. HDF5 with one group per element (named as in the BED file) and one dataset of shape
`(element length, 4)` per output track, columns in ACGT order:

```
/ALDH1A1_enh1/ATAC
/ALDH1A1_enh1/RNA_ALDH1A1
/ALDH1A1_enh1/TF_GATA1
```

Values are attributions projected onto the reference base, so each row has one non-zero entry at the reference nucleotide. Read it with `element_epistasis.ism.load_ism_dict`, which returns track names with a space (`RNA ALDH1A1`). Elements are indexed by their alphabetical position in this file, and that index (`seq_idx`) is reused in every prediction table below.

### Prediction tables

All prediction tables share one layout: one row per predicted sequence, with the unperturbed reference once (`n_scrambled = 0`, `shuffle_idx` missing) and every perturbation once per dinucleotide shuffle (`shuffle_idx` 0 to 9). The `_raw` tables keep this replicate structure and are what the analysis reads; the tables without `_raw` are averaged over shuffles.

Columns common to all of them:

| Column | Type | Content |
|---|---|---|
| `name` | str | unit of analysis: the gene (enhancer level, cross-enhancer) or the element (within-enhancer) |
| `combo_id`, `combo_label` | int, str | identifier of the perturbation |
| `n_scrambled` | int | number of units scrambled, 0 for the reference |
| `scrambled_seqlets` | list[str] | units scrambled in this row |
| `unscrambled_seqlets` | list[str] | units left intact |
| `shuffle_idx` | float | shuffle replicate, missing for the reference |
| `RNA_counts` | float or list[float] | predicted expression of the target gene(s) |
| `ATAC_counts` | float or list[float] | predicted accessibility, summed over the element(s) |

Specific to each table:

| File | Unit scrambled | Readouts | Extra columns |
|---|---|---|---|
| `enhancer_insilico_KO.parquet` | whole element (`GENE_LABEL`) | `RNA_counts` scalar; `ATAC_counts` list, one per element of the gene | `target_enhancer`: element order of `ATAC_counts`; `seq_idx` |
| `TF_insilico_KO_raw.parquet` | seqlet within one element (`GATA1`, `FOS#2` for a repeat) | `ATAC_counts` scalar over that element; `RNA_counts` list | `seq_idx`, `chrom`, `start`, `end`, `target_gene`, `sequence` (the element sequence) |
| `TF_insilico_KO_controls_raw.parquet` | control window (`ctrl#1`) | as above | as above |
| `TF_cross_enhancer_raw.parquet` | seqlet labelled with its element (`ALDH1A1_enh1\|ATF4`) | `ATAC_counts` list, one per element of the gene; `RNA_counts` list | `enh_keys`, `target_enhancer`: element order; `pair_type` (`baseline`, `single`, `within`, `across`, `all`); `n_enhancers_hit` |

Readouts must be strictly positive: they are modelled with a Gamma GLM on the log link. The enhancer-level table is stored without the 1 Mb input sequences, which are only needed to submit predictions.

### Derived tables

| File | Content |
|---|---|
| `enhancer_insilico_single_KO_main_effects.csv` | enhancer-level GLM main effects; notebook 02 ranks elements by them when selecting cross-enhancer seqlets |
| `enhancer_insilico_KO_inters.csv` | enhancer-level GLM interactions; compared with the aggregated cross-enhancer motif interactions |

Both are written by notebook 01, so run it before notebook 02.

## `reference/`: not version controlled

Only needed to generate new predictions (`RUN_PREDICTIONS = True`). Set `AGEE_REFERENCE` to use an existing copy instead.

| File | Content |
|---|---|
| `genome.fa`, `genome.fa.fai` | hg38 FASTA and index. Without it, notebook 02 reads the element sequences from the ISM tracks. |
| `genome.chrom.sizes` | chromosome lengths, used to check the prediction windows |
| `gencode.v46.annotation.gtf.gz.feather` | downloaded automatically on first use |
