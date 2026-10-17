"""Repository paths, data locations and the AlphaGenome API key.

Environment overrides: AGEE_ROOT (repository root), AGEE_REFERENCE (hg38 FASTA,
chromosome sizes, GTF cache) and ALPHAGENOME_API_KEY.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("AGEE_ROOT", Path(__file__).resolve().parents[2]))

DATA = ROOT / "data"
RAW = DATA / "raw"
META = DATA / "meta"
PROCESSED = DATA / "processed"
REFERENCE = Path(os.environ.get("AGEE_REFERENCE", DATA / "reference"))

RESULTS = ROOT / "results"
SUMMARIES = RESULTS / "summaries"
FIGURES = RESULTS / "figures"

HG38_FASTA = REFERENCE / "genome.fa"
CHROM_SIZES = REFERENCE / "genome.chrom.sizes"

JASPAR_MEME = META / "JASPAR2026_CORE_vertebrates_non-redundant_pfms_meme.txt"

GTF_URL = ("https://storage.googleapis.com/alphagenome/reference/gencode/"
           "hg38/gencode.v46.annotation.gtf.gz.feather")
GTF_CACHE = REFERENCE / "gencode.v46.annotation.gtf.gz.feather"

# EFO term for K562, used when the track metadata cannot be queried because no
# API key is available. It is the curie the predictions were generated with.
K562_CURIE = "EFO:0002067"


def alphagenome_api_key() -> str | None:
    """Read the AlphaGenome API key.

    Returns:
        The key from ALPHAGENOME_API_KEY or ~/.alphagenome_api_key, or None if
        neither is set.
    """
    key = os.environ.get("ALPHAGENOME_API_KEY")
    if key:
        return key.strip()
    key_file = Path.home() / ".alphagenome_api_key"
    if key_file.exists():
        return key_file.read_text().strip()
    return None


def load_gtf() -> pd.DataFrame:
    """Load the GENCODE v46 annotation used by AlphaGenome.

    Downloads the feather file into the reference folder on first use and reads
    the cached copy afterwards.

    Returns:
        The annotation as a DataFrame.
    """
    if GTF_CACHE.exists():
        return pd.read_feather(GTF_CACHE)
    gtf = pd.read_feather(GTF_URL)
    GTF_CACHE.parent.mkdir(parents=True, exist_ok=True)
    gtf.to_feather(GTF_CACHE)
    return gtf


def ensure_output_dirs() -> None:
    """Create the processed, summary and figure folders if missing."""
    for folder in (PROCESSED, SUMMARIES, FIGURES):
        folder.mkdir(parents=True, exist_ok=True)
