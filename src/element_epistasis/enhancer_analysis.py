"""Tables for the enhancer-level in silico knockout.

Reshapes and annotates GLM coefficient tables, validates them against CRISPRi,
and computes model-free contrasts. With a log link the GLM interaction b_ij
equals I = delta_ij - delta_i - delta_j (negative: super-additive).
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

_TERM_SPLIT = re.compile(r"\s*:\s*")


def normalise_interaction_table(
    df: pd.DataFrame,
    gene_col: str = "gene",
    term_col: str = "term",
    a_col: str = "enhA",
    b_col: str = "enhB",
) -> pd.DataFrame:
    """Bring an interaction table into one layout, from either a term or an enhA/enhB column layout.

    Args:
        df: Interaction table.
        gene_col: Gene column.
        term_col: Colon-joined term column, if present.
        a_col: First fully qualified enhancer column, if present.
        b_col: Second fully qualified enhancer column, if present.

    Returns:
        Copy of df with full_a, full_b and term.

    Raises:
        ValueError: If neither layout is found.
    """
    out = df.copy()
    has_pair = a_col in out.columns and b_col in out.columns
    has_term = term_col in out.columns

    if has_pair:
        out["full_a"] = out[a_col].astype(str)
        out["full_b"] = out[b_col].astype(str)
        if not has_term:
            # Recover the short term name by stripping the gene prefix.
            def _short(row: pd.Series, col: str) -> str:
                value = row[col]
                prefix = f"{row[gene_col]}_"
                return value[len(prefix):] if value.startswith(prefix) else value

            out[term_col] = out.apply(
                lambda r: f"{_short(r, 'full_a')}:{_short(r, 'full_b')}", axis=1
            )

    elif has_term:
        parts = out[term_col].astype(str).apply(lambda t: _TERM_SPLIT.split(t))
        keep = parts.apply(len) == 2
        if (~keep).any():
            warnings.warn(
                f"Dropping {int((~keep).sum())} term(s) that are not pairwise."
            )
        out = out[keep].copy()
        parts = parts[keep]
        out["full_a"] = out[gene_col] + "_" + parts.str[0]
        out["full_b"] = out[gene_col] + "_" + parts.str[1]

    else:
        raise ValueError(
            f"Cannot recognise this table: expected either a '{term_col}' "
            f"column or both '{a_col}' and '{b_col}'. "
            f"Columns present: {list(df.columns)}"
        )

    return out.reset_index(drop=True)


def split_coef_table(
    coef_df: pd.DataFrame,
    term_col: str = "term",
    gene_col: str = "gene",
    drop_intercept: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a concatenated GLM coefficient table into main effects and interactions.

    Args:
        coef_df: Coefficient tables of all genes, one row per term.
        term_col: Term column.
        gene_col: Gene column.
        drop_intercept: Drop the intercept rows.

    Returns:
        Tuple of (main effects with full_a, interactions with full_a and full_b),
        using fully qualified "GENE_enhN" names.

    Raises:
        ValueError: If term_col is missing.
    """
    if term_col not in coef_df.columns:
        raise ValueError(
            f"No '{term_col}' column. This table looks like it has already "
            f"been reshaped; pass it to normalise_interaction_table instead. "
            f"Columns present: {list(coef_df.columns)}"
        )

    df = coef_df.copy()
    if drop_intercept:
        df = df[~df[term_col].str.contains("intercept", case=False, na=False)]

    is_interaction = df[term_col].str.contains(":", na=False)
    mains = df[~is_interaction].copy()
    inters = df[is_interaction].copy()

    mains["full_a"] = mains[gene_col] + "_" + mains[term_col]

    if not inters.empty:
        inters = normalise_interaction_table(inters, gene_col=gene_col,
                                             term_col=term_col)

    return mains.reset_index(drop=True), inters.reset_index(drop=True)


def attach_main_effects(
    inters: pd.DataFrame,
    coef_df: pd.DataFrame,
    gene_col: str = "gene",
    term_col: str = "term",
    coef_col: str = "coef",
    intercept_name: str = "intercept",
    exclude_terms: tuple[str, ...] = ("alpha",),
    a_col: str = "full_a",
    b_col: str = "full_b",
) -> pd.DataFrame:
    """Add the two main effects and the baseline exp(intercept) to each interaction.

    Args:
        inters: Interactions with fully qualified enhancer names.
        coef_df: Full coefficient table including the intercept (not split).
        gene_col: Gene column.
        term_col: Term column.
        coef_col: Coefficient column.
        intercept_name: Term name of the intercept.
        exclude_terms: Terms that are not enhancer effects, e.g. the NB "alpha".
        a_col: First enhancer column in inters.
        b_col: Second enhancer column in inters.

    Returns:
        Copy of inters with main_a, main_b and baseline; missing where not found.
    """
    table = coef_df.copy()

    is_intercept = table[term_col].astype(str).str.lower() == intercept_name.lower()
    is_excluded = table[term_col].isin(exclude_terms)
    is_interaction = table[term_col].astype(str).str.contains(":", na=False)

    mains = table[~is_intercept & ~is_excluded & ~is_interaction].copy()
    mains["_full"] = mains[gene_col] + "_" + mains[term_col].astype(str)
    lookup = mains.drop_duplicates("_full").set_index("_full")[coef_col]

    out = inters.copy()
    out["main_a"] = out[a_col].map(lookup)
    out["main_b"] = out[b_col].map(lookup)

    # Log link: the linear predictor is the intercept when nothing is scrambled.
    intercepts = table[is_intercept].drop_duplicates(gene_col)
    baseline = intercepts.set_index(gene_col)[coef_col].apply(np.exp)
    out["baseline"] = out[gene_col].map(baseline)

    missing_main = int(out[["main_a", "main_b"]].isna().any(axis=1).sum())
    if missing_main:
        warnings.warn(
            f"{missing_main} pair(s) could not be matched to both main effects."
        )
    if out["baseline"].isna().all():
        warnings.warn(
            f"No term named '{intercept_name}' found; baseline not attached. "
            f"Pass the unsplit coefficient table, which still has the intercept."
        )
    return out


def annotate_pair_distance(
    pairs: pd.DataFrame,
    enhancers: pd.DataFrame,
    a_col: str = "full_a",
    b_col: str = "full_b",
    name_col: str = "name",
    start_col: str = "start",
    end_col: str = "end",
    chrom_col: str = "chrom",
) -> pd.DataFrame:
    """Add the genomic distance between the midpoints of the two enhancers of each pair.

    Args:
        pairs: Table with two enhancer name columns.
        enhancers: Enhancer coordinates, one row per enhancer.
        a_col: First enhancer column.
        b_col: Second enhancer column.
        name_col: Name column in enhancers.
        start_col: Start column.
        end_col: End column.
        chrom_col: Chromosome column.

    Returns:
        Copy of pairs with distance_bp and same_chrom; distance is missing across
        chromosomes.
    """
    coords = enhancers[[name_col, chrom_col, start_col, end_col]].drop_duplicates(
        subset=[name_col]
    )
    coords = coords.assign(
        _mid=(coords[start_col] + coords[end_col]) / 2.0
    )[[name_col, chrom_col, "_mid"]]

    out = pairs.copy()
    for side, col in [("a", a_col), ("b", b_col)]:
        renamed = coords.rename(columns={
            name_col: col, chrom_col: f"chrom_{side}", "_mid": f"mid_{side}",
        })
        out = out.merge(renamed, on=col, how="left")

    out["same_chrom"] = out["chrom_a"] == out["chrom_b"]
    distance = (out["mid_a"] - out["mid_b"]).abs()
    out["distance_bp"] = distance.where(out["same_chrom"])
    return out


def _proportion_ci(successes: int, total: int,
                   method: str = "wilson") -> tuple[float, float, float]:
    """Wilson interval and two-sided binomial p-value against 0.5 for a proportion."""
    if total == 0:
        return np.nan, np.nan, np.nan
    test = stats.binomtest(successes, total, 0.5)
    lo, hi = test.proportion_ci(method=method)
    return float(lo), float(hi), float(test.pvalue)


def _concordance_block(df: pd.DataFrame, insilico_col: str,
                       measured_col: str, prefix: str) -> dict:
    """Sign concordance, interval and binomial test for one subset, with keys suffixed by prefix."""
    n = len(df)
    if n == 0:
        return {f"n{prefix}": 0, f"n_concordant{prefix}": 0,
                f"concordance{prefix}": np.nan, f"ci_lo{prefix}": np.nan,
                f"ci_hi{prefix}": np.nan, f"binom_p{prefix}": np.nan}
    concordant = int((np.sign(df[insilico_col])
                      == np.sign(df[measured_col])).sum())
    lo, hi, pval = _proportion_ci(concordant, n)
    return {f"n{prefix}": n, f"n_concordant{prefix}": concordant,
            f"concordance{prefix}": concordant / n, f"ci_lo{prefix}": lo,
            f"ci_hi{prefix}": hi, f"binom_p{prefix}": pval}


def summarise_validation(
    comp: pd.DataFrame,
    insilico_col: str = "coef",
    measured_col: str = "interaction_beta",
    insilico_p_col: str | None = None,
    alpha: float = 0.05,
) -> dict:
    """Summarise agreement in sign and magnitude between in silico and measured coefficients.

    Reported for all effects and for effects significant in silico; the screen is
    not used to define subsets.

    Args:
        comp: Merged in silico and measured coefficients.
        insilico_col: In silico coefficient column.
        measured_col: Measured coefficient column.
        insilico_p_col: In silico significance column; the second subset is
            empty when None.
        alpha: Significance threshold.

    Returns:
        Dictionary of counts, concordance, Wilson intervals, binomial p-values and
        correlations, with keys suffixed "" and "_insilico".
    """
    df = comp.dropna(subset=[insilico_col, measured_col])
    out = {"alpha": alpha}
    out.update(_concordance_block(df, insilico_col, measured_col, ""))
 
    if len(df) == 0:
        return out
 
    # Correlation and calibration on the unrestricted set.
    if df[insilico_col].nunique() > 1 and df[measured_col].nunique() > 1:
        out["pearson_r"], out["pearson_p"] = stats.pearsonr(
            df[measured_col], df[insilico_col])
        out["spearman_rho"], out["spearman_p"] = stats.spearmanr(
            df[measured_col], df[insilico_col])
        out["slope"] = stats.linregress(df[measured_col],
                                        df[insilico_col]).slope
 
    insilico_ok = (df[insilico_p_col] < alpha
                   if insilico_p_col and insilico_p_col in df.columns
                   else pd.Series(False, index=df.index))
 
    out.update(_concordance_block(df[insilico_ok], insilico_col,
                                  measured_col, "_insilico"))
    return out

def summarise_trans_effects(
    interdependency: pd.DataFrame,
    by: str = "gene",
    value_col: str = "delta",
) -> pd.DataFrame:
    """Average the trans (off-diagonal) effects of an interdependency table.

    Args:
        interdependency: Output of build_interdependency.
        by: "gene" for one row per gene, "perturbed" for one row per perturbed
            enhancer.
        value_col: Effect column.

    Returns:
        One row per group with mean and median trans effect and cell count.
    """
    trans = interdependency[~interdependency["is_cis"]]
    if trans.empty:
        return pd.DataFrame(columns=[by, "mean_trans", "median_trans", "n_trans"])

    out = (trans.groupby(by)[value_col]
                .agg(mean_trans="mean", median_trans="median", n_trans="size")
                .reset_index())
    return out


def summarise_pair_trans(
    interdependency: pd.DataFrame,
    gene_col: str = "gene",
    perturbed_col: str = "perturbed",
    measured_col: str = "measured",
    value_col: str = "delta",
) -> pd.DataFrame:
    """Summarise the two directed trans effects of each enhancer pair.

    Args:
        interdependency: Output of build_interdependency.
        gene_col: Gene column.
        perturbed_col: Perturbed enhancer column.
        measured_col: Measured locus column.
        value_col: Effect column.

    Returns:
        One row per unordered pair (enhA, enhB sorted) with trans_strongest (more
        negative direction), trans_asymmetry (absolute difference) and trans_mean.
    """
    trans = interdependency[~interdependency["is_cis"]].copy()
    if trans.empty:
        return pd.DataFrame(columns=[gene_col, "enhA", "enhB", "trans_strongest",
                                     "trans_weakest", "trans_mean",
                                     "trans_asymmetry", "n_directions"])

    # Sort the pair so that both directed cells land in the same group.
    ordered = trans.apply(
        lambda r: tuple(sorted([str(r[perturbed_col]), str(r[measured_col])])),
        axis=1,
    )
    trans["enhA"] = [pair[0] for pair in ordered]
    trans["enhB"] = [pair[1] for pair in ordered]

    records = []
    for (gene, a, b), group in trans.groupby([gene_col, "enhA", "enhB"]):
        values = group[value_col].dropna().to_numpy()
        if not len(values):
            continue
        records.append({
            gene_col: gene,
            "enhA": a,
            "enhB": b,
            "trans_strongest": float(np.min(values)),
            "trans_weakest": float(np.max(values)),
            "trans_mean": float(np.mean(values)),
            "trans_asymmetry": float(np.max(values) - np.min(values))
            if len(values) > 1 else np.nan,
            "n_directions": int(len(values)),
        })

    return pd.DataFrame(records)


def join_pair_trans(
    comp: pd.DataFrame,
    pair_trans: pd.DataFrame,
    a_col: str = "enhA",
    b_col: str = "enhB",
) -> pd.DataFrame:
    """Attach per-pair trans statistics to a comparison table, joining on the sorted pair.

    Args:
        comp: Table with two enhancer columns.
        pair_trans: Output of summarise_pair_trans.
        a_col: First enhancer column in comp.
        b_col: Second enhancer column in comp.

    Returns:
        Copy of comp with the trans columns, missing where unmatched.
    """
    if comp.empty or pair_trans.empty:
        return comp.copy()

    def key(frame: pd.DataFrame, first: str, second: str) -> pd.DataFrame:
        pairs = frame.apply(
            lambda r: tuple(sorted([str(r[first]), str(r[second])])), axis=1)
        return pd.DataFrame({"_a": [p[0] for p in pairs],
                             "_b": [p[1] for p in pairs]}, index=frame.index)

    left = comp.join(key(comp, a_col, b_col))
    right = pair_trans.join(key(pair_trans, "enhA", "enhB"))
    keep = ["_a", "_b", "trans_strongest", "trans_weakest", "trans_mean",
            "trans_asymmetry", "n_directions"]
    keep = [c for c in keep if c in right.columns]

    merged = left.merge(right[keep], on=["_a", "_b"], how="left")
    return merged.drop(columns=["_a", "_b"])


def _is_vector_column(series: pd.Series) -> bool:
    """Whether a column holds lists or arrays rather than scalars."""
    for value in series:
        if value is None:
            continue
        if isinstance(value, (list, tuple, np.ndarray)):
            return True
        return False
    return False


def expand_locus_values(
    scram: pd.DataFrame,
    value_col: str = "ATAC_counts",
    locus_col: str = "target_enhancer",
    out_col: str | None = None,
    locus_name: str = "locus",
) -> pd.DataFrame:
    """Explode a per-locus list readout into one row per measured locus.

    Args:
        scram: Scramble results with a list readout column.
        value_col: Column with the per-locus values.
        locus_col: Column with the aligned locus names.
        out_col: Name of the scalar output column; defaults to value_col.
        locus_name: Name of the new locus column.

    Returns:
        Long copy of scram with a scalar readout and a locus column.

    Raises:
        ValueError: If values and locus names differ in length.
    """
    out_col = out_col or value_col
    records = []

    for _, row in scram.iterrows():
        values = row[value_col]
        loci = row[locus_col]

        values = list(values) if isinstance(values, (list, tuple, np.ndarray)) \
            else [values]
        loci = list(loci) if isinstance(loci, (list, tuple, np.ndarray)) \
            else [loci]

        if len(values) != len(loci):
            raise ValueError(
                f"Row {row.name}: '{value_col}' has {len(values)} value(s) but "
                f"'{locus_col}' names {len(loci)} locus/loci. These must align."
            )

        for locus, value in zip(loci, values):
            record = row.to_dict()
            record[out_col] = float(value)
            record[locus_name] = locus
            records.append(record)

    return pd.DataFrame(records).reset_index(drop=True)


def aggregate_locus_values(
    scram: pd.DataFrame,
    value_col: str = "ATAC_counts",
    locus_col: str = "target_enhancer",
    how: str = "sum",
    out_col: str | None = None,
) -> pd.DataFrame:
    """Collapse a per-locus list readout to one scalar per row.

    Args:
        scram: Scramble results with a list readout column.
        value_col: Column with the per-locus values.
        locus_col: Column with the aligned locus names (kept for symmetry with
            expand_locus_values).
        how: "sum", "mean" or "max"; sum and mean give identical log contrasts.
        out_col: Name of the scalar output column; defaults to value_col.

    Returns:
        Copy of scram with a scalar readout column.
    """
    if how not in {"sum", "mean", "max"}:
        raise ValueError(f"Unknown aggregation '{how}'; use sum, mean or max.")

    reducer = dict(sum=np.sum, mean=np.mean, max=np.max)[how]
    out_col = out_col or value_col
    out = scram.copy()

    def _reduce(value: Any) -> float:
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value, dtype=float).ravel()
            return float(reducer(arr)) if arr.size else np.nan
        return float(value)

    out[out_col] = out[value_col].apply(_reduce)
    return out


def _require_scalar_column(scram: pd.DataFrame, value_col: str) -> None:
    """Raise a clear error when the readout column holds lists."""
    if value_col in scram.columns and _is_vector_column(scram[value_col]):
        raise ValueError(
            f"'{value_col}' holds one value per measured locus, not a scalar. "
            f"Choose how to handle that first:\n"
            f"  expand_locus_values(scram, '{value_col}')  then pass "
            f"name_col='locus' for a score per locus, or\n"
            f"  aggregate_locus_values(scram, '{value_col}', how='sum')  for "
            f"one gene-level score.\n"
            f"See the docstrings for which is appropriate."
        )


def _combination_key(value: Any) -> tuple:
    """Hashable key for a set of scrambled elements."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return tuple(sorted(str(v) for v in value))
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ()
    return (str(value),)


def _combination_means(
    group: pd.DataFrame,
    value_col: str,
    scrambled_col: str,
    transform: Callable[[float], float],
) -> dict[tuple, float]:
    """Mean transformed readout per scramble combination, averaging shuffle replicates first."""
    keys = group[scrambled_col].apply(_combination_key)
    values = group[value_col].astype(float).apply(transform)
    return values.groupby(keys).mean().to_dict()


def build_pairwise_epistasis(
    scram: pd.DataFrame,
    value_col: str = "RNA",
    name_col: str = "name",
    scrambled_col: str = "scrambled_seqlets",
    n_scrambled_col: str = "n_scr",
    shuffle_col: str = "shuffle_idx",
    scale: str = "log",
    pseudocount: float = 1e-6,
) -> pd.DataFrame:
    """Model-free pairwise interaction I = delta_AB - delta_A - delta_B per gene and enhancer pair.

    Shuffles are averaged before the contrast; when shuffle_col is present the
    contrast is also paired by shuffle to give a standard error and p-value.

    Args:
        scram: Scramble results, one row per sequence.
        value_col: Scalar readout column.
        name_col: Gene column.
        scrambled_col: Column with the scrambled enhancers.
        n_scrambled_col: Column with the number scrambled.
        shuffle_col: Shuffle index column, if present.
        scale: "log" or "linear".
        pseudocount: Added before taking the log.

    Returns:
        One row per gene and pair with the interaction, main effects, baseline and,
        where estimable, standard error, replicate count and p-value.
    """
    _require_scalar_column(scram, value_col)

    def transform(x: float) -> float:
        return float(np.log(float(x) + pseudocount)) if scale == "log" else float(x)

    records = []

    for name, group in scram.groupby(name_col, sort=True):
        means = _combination_means(group, value_col, scrambled_col, transform)
        if () not in means:
            continue
        y0 = means[()]

        singles = {key[0]: value - y0
                   for key, value in means.items() if len(key) == 1}
        pairs = {key: value for key, value in means.items() if len(key) == 2}

        has_shuffles = shuffle_col in group.columns
        if has_shuffles:
            keys = group[scrambled_col].apply(_combination_key)
            by_combo = {
                key: sub.set_index(shuffle_col)[value_col].astype(float).apply(transform)
                for key, sub in group.groupby(keys)
            }
        else:
            by_combo = {}

        for key, pair_mean in pairs.items():
            a, b = key
            if a not in singles or b not in singles:
                continue

            delta_ab = pair_mean - y0
            interaction = delta_ab - singles[a] - singles[b]

            sem, pval, n_shuffles = np.nan, np.nan, np.nan
            if has_shuffles and all(k in by_combo for k in [key, (a,), (b,)]):
                joined = pd.concat(
                    {"ab": by_combo[key], "a": by_combo[(a,)], "b": by_combo[(b,)]},
                    axis=1,
                ).dropna()
                if len(joined) > 1:
                    per_shuffle = (joined["ab"] - joined["a"] - joined["b"] + y0)
                    n_shuffles = len(per_shuffle)
                    sem = float(stats.sem(per_shuffle))
                    if n_shuffles > 2:
                        pval = float(stats.ttest_1samp(per_shuffle, 0.0).pvalue)

            records.append(dict(
                enhancer=name,
                tfA=a,
                tfB=b,
                interaction=float(interaction),
                sem=sem,
                n_shuffles=n_shuffles,
                pval=pval,
                delta_a=float(singles[a]),
                delta_b=float(singles[b]),
                delta_ab=float(delta_ab),
                baseline=float(np.exp(y0) if scale == "log" else y0),
                readout=value_col,
                scale=scale,
            ))

    return pd.DataFrame(records)


def build_coop_table(
    scram: pd.DataFrame,
    value_col: str = "RNA",
    name_col: str = "name",
    scrambled_col: str = "scrambled_seqlets",
    n_scrambled_col: str = "n_scr",
    shuffle_col: str = "shuffle_idx",
    scale: str = "log",
    pseudocount: float = 1e-6,
) -> pd.DataFrame:
    """Per-gene cooperativity: delta_all minus the sum of single-enhancer effects.

    Args:
        scram: Scramble results, one row per sequence.
        value_col: Scalar readout column.
        name_col: Gene column.
        scrambled_col: Column with the scrambled enhancers.
        n_scrambled_col: Column with the number scrambled.
        shuffle_col: Shuffle index column, if present.
        scale: "log" or "linear".
        pseudocount: Added before taking the log.

    Returns:
        One row per gene with additive_prediction, observed and cooperativity
        (sum_d_rna and d_rna_all are legacy aliases).
    """
    _require_scalar_column(scram, value_col)

    def transform(x: float) -> float:
        return float(np.log(float(x) + pseudocount)) if scale == "log" else float(x)

    rows = []

    for name, group in scram.groupby(name_col, sort=True):
        means = _combination_means(group, value_col, scrambled_col, transform)
        if () not in means:
            continue
        y0 = means[()]

        singles = {key[0]: value - y0
                   for key, value in means.items() if len(key) == 1}
        if not singles:
            continue

        # The all-scrambled state is the combination touching the most enhancers.
        n_max = max(len(key) for key in means)
        if n_max < 2:
            continue
        all_keys = [key for key in means if len(key) == n_max]
        delta_all = float(np.mean([means[key] for key in all_keys])) - y0

        additive = float(np.sum(list(singles.values())))
        rows.append(dict(
            name=name,
            n_seqlets=int(n_max),
            n_singles=len(singles),
            additive_prediction=additive,
            observed=float(delta_all),
            cooperativity=float(delta_all - additive),
            readout=value_col,
            scale=scale,
            # Legacy aliases so existing notebook cells keep working.
            sum_d_rna=additive,
            d_rna_all=float(delta_all),
        ))

    return pd.DataFrame(rows)


def build_interdependency(
    long: pd.DataFrame,
    value_col: str = "ATAC_counts",
    gene_col: str = "name",
    locus_col: str = "locus",
    scrambled_col: str = "scrambled_seqlets",
    shuffle_col: str = "shuffle_idx",
    scale: str = "log",
    pseudocount: float = 1e-6,
) -> pd.DataFrame:
    """Effect of scrambling each enhancer on the accessibility of every enhancer of the gene.

    Args:
        long: Output of expand_locus_values.
        value_col: Scalar readout column.
        gene_col: Gene column.
        locus_col: Measured locus column.
        scrambled_col: Column with the scrambled enhancers.
        shuffle_col: Shuffle index column, if present.
        scale: "log" or "linear".
        pseudocount: Added before taking the log.

    Returns:
        One row per perturbed enhancer and measured locus with the effect,
        standard error, p-value where estimable and is_cis.
    """
    _require_scalar_column(long, value_col)

    def transform(x: float) -> float:
        return float(np.log(float(x) + pseudocount)) if scale == "log" else float(x)

    records = []

    for (gene, locus), group in long.groupby([gene_col, locus_col], sort=True):
        keys = group[scrambled_col].apply(_combination_key)
        values = group[value_col].astype(float).apply(transform)

        means = values.groupby(keys).mean().to_dict()
        if () not in means:
            continue
        y0 = means[()]

        has_shuffles = shuffle_col in group.columns
        by_combo = {
            key: sub.set_index(shuffle_col)[value_col].astype(float).apply(transform)
            for key, sub in group.groupby(keys)
        } if has_shuffles else {}

        for key, mean_value in means.items():
            if len(key) != 1:
                continue
            perturbed = key[0]

            sem, pval, n_shuffles = np.nan, np.nan, np.nan
            series = by_combo.get(key)
            if series is not None and len(series.dropna()) > 1:
                per_shuffle = series.dropna() - y0
                n_shuffles = len(per_shuffle)
                sem = float(stats.sem(per_shuffle))
                if n_shuffles > 2:
                    pval = float(stats.ttest_1samp(per_shuffle, 0.0).pvalue)

            records.append(dict(
                gene=gene,
                perturbed=perturbed,
                measured=locus,
                is_cis=perturbed == locus,
                delta=float(mean_value - y0),
                sem=sem,
                n_shuffles=n_shuffles,
                pval=pval,
                baseline=float(np.exp(y0) if scale == "log" else y0),
                readout=value_col,
                scale=scale,
            ))

    return pd.DataFrame(records)


def build_model_comparison(
    results_by_setting: dict[str, dict],
    verbose: bool = True,
) -> pd.DataFrame:
    """Tabulate fit diagnostics across model families and interaction orders.

    Args:
        results_by_setting: Mapping of a label such as "gamma, order 2" to the
            fit_all_genes output for that setting.
        verbose: Report which diagnostics are available.

    Returns:
        One row per setting and gene with convergence, number of terms and mean
        standard error (AIC and deviance when available).
    """
    rows = []
    for label, results in results_by_setting.items():
        for gene, fit in results.items():
            table = fit.get("coef_table")
            # A failed fit stores coef_table=None, so every access must be guarded.
            has_table = isinstance(table, pd.DataFrame) and not table.empty
            has_se = has_table and "se" in table.columns and table["se"].notna().any()

            rows.append(dict(
                setting=label,
                gene=gene,
                converged=bool(fit.get("converged", True)),
                family=fit.get("family"),
                max_order=fit.get("max_order"),
                n_obs=fit.get("n_obs", np.nan),
                n_terms=int(len(table)) if has_table else np.nan,
                mean_se=float(table["se"].mean(skipna=True)) if has_se else np.nan,
                aic=float(fit["aic"]) if fit.get("aic") is not None else np.nan,
                deviance=float(fit["deviance"])
                if fit.get("deviance") is not None else np.nan,
            ))

    out = pd.DataFrame(rows)
    if verbose and not out.empty:
        available = [c for c in ["mean_se", "n_terms", "aic", "deviance"]
                     if out[c].notna().any()]
        missing = [c for c in ["aic", "deviance"] if c not in available]
        print(f"Populated diagnostics: {', '.join(available) or 'none'}")
        if missing:
            print(f"Not available from this fitter: {', '.join(missing)}. "
                  f"See build_model_comparison docstring.")
        failures = out.loc[~out["converged"], "setting"].value_counts()
        for setting, count in failures.items():
            print(f"{setting}: {count} of "
                  f"{int((out['setting'] == setting).sum())} fits did not converge.")
    return out