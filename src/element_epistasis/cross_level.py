"""Link motif-level interactions to the enhancer level and to the CRISPRi screen.

Contents: measured-table normalisation; motif-pair to enhancer-pair links
(link_motif_pairs_to_enhancers, summarise_pair_link, summarise_within_gene_link);
motif content to enhancer knockout effect (aggregate_motifs_per_enhancer,
link_motif_effects_to_enhancers, summarise_motif_enhancer_link); spacing versus
saturation (summarise_spacing_partial); text rendering (format_cross_level_summary).
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .tf_cooperativity import partial_correlation

# Gene prefix and enhancer suffix of a fully qualified identifier such as
# ``MGAT1_enh2-3``. Non-greedy on the gene so that the first ``_enh`` wins.
_QUALIFIED = re.compile(r"^(?P<gene>.+?)_(?P<enh>enh.*)$")


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _pair_key(frame: pd.DataFrame, a_col: str, b_col: str) -> pd.DataFrame:
    """Order-independent join key (_key_a, _key_b) for a pair of enhancer columns."""
    pairs = [tuple(sorted([str(a), str(b)]))
             for a, b in zip(frame[a_col], frame[b_col])]
    return pd.DataFrame({"_key_a": [p[0] for p in pairs],
                         "_key_b": [p[1] for p in pairs]}, index=frame.index)


def _corr_stats(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray,
                prefix: str = "") -> dict:
    """Pearson and Spearman correlation on the shared finite values, with keys prefixed."""
    frame = pd.DataFrame({
        "x": pd.to_numeric(pd.Series(x).reset_index(drop=True), errors="coerce"),
        "y": pd.to_numeric(pd.Series(y).reset_index(drop=True), errors="coerce"),
    }).dropna()

    out = {f"{prefix}n": len(frame)}
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        out.update({f"{prefix}pearson_r": np.nan, f"{prefix}pearson_p": np.nan,
                    f"{prefix}spearman_rho": np.nan,
                    f"{prefix}spearman_p": np.nan})
        return out

    r, pr = stats.pearsonr(frame["x"], frame["y"])
    rho, prho = stats.spearmanr(frame["x"], frame["y"])
    out.update({f"{prefix}pearson_r": float(r), f"{prefix}pearson_p": float(pr),
                f"{prefix}spearman_rho": float(rho),
                f"{prefix}spearman_p": float(prho)})
    return out


def _sign_concordance(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray,
                      prefix: str = "") -> dict:
    """Sign agreement with a Wilson interval and a two-sided binomial test against 0.5."""
    frame = pd.DataFrame({
        "x": pd.to_numeric(pd.Series(x).reset_index(drop=True), errors="coerce"),
        "y": pd.to_numeric(pd.Series(y).reset_index(drop=True), errors="coerce"),
    }).dropna()
    n = len(frame)
    if n == 0:
        return {f"{prefix}n_sign": 0, f"{prefix}n_concordant": 0,
                f"{prefix}concordance": np.nan, f"{prefix}ci_lo": np.nan,
                f"{prefix}ci_hi": np.nan, f"{prefix}binom_p": np.nan}

    concordant = int((np.sign(frame["x"]) == np.sign(frame["y"])).sum())
    test = stats.binomtest(concordant, n, 0.5)
    lo, hi = test.proportion_ci(method="wilson")
    return {f"{prefix}n_sign": n, f"{prefix}n_concordant": concordant,
            f"{prefix}concordance": concordant / n, f"{prefix}ci_lo": float(lo),
            f"{prefix}ci_hi": float(hi), f"{prefix}binom_p": float(test.pvalue)}


def _jackknife_influence(x: pd.Series | np.ndarray,
                         y: pd.Series | np.ndarray) -> dict:
    """Leave-one-out range of the Pearson correlation and the most influential point."""
    frame = pd.DataFrame({
        "x": pd.to_numeric(pd.Series(x).reset_index(drop=True), errors="coerce"),
        "y": pd.to_numeric(pd.Series(y).reset_index(drop=True), errors="coerce"),
    }).dropna()

    if len(frame) < 5:
        return {"jack_n": len(frame), "jack_r_full": np.nan,
                "jack_r_min": np.nan, "jack_r_max": np.nan,
                "jack_max_shift": np.nan, "jack_worst_index": None}

    full = stats.pearsonr(frame["x"], frame["y"])[0]
    values = []
    for i in frame.index:
        sub = frame.drop(index=i)
        if sub["x"].nunique() < 2 or sub["y"].nunique() < 2:
            continue
        values.append((stats.pearsonr(sub["x"], sub["y"])[0], i))

    if not values:
        return {"jack_n": len(frame), "jack_r_full": float(full),
                "jack_r_min": np.nan, "jack_r_max": np.nan,
                "jack_max_shift": np.nan, "jack_worst_index": None}

    r_values = np.array([v[0] for v in values])
    shifts = np.abs(r_values - full)
    worst = values[int(np.argmax(shifts))][1]
    return {"jack_n": len(frame), "jack_r_full": float(full),
            "jack_r_min": float(r_values.min()),
            "jack_r_max": float(r_values.max()),
            "jack_max_shift": float(shifts.max()),
            "jack_worst_index": worst}


# --------------------------------------------------------------------------- #
# Normalising the measured CRISPRi tables
# --------------------------------------------------------------------------- #

def _qualify(gene: str, unit: str) -> str:
    """Prefix a bare enhancer name with its gene, leaving qualified names unchanged."""
    unit = str(unit)
    return unit if unit.startswith(f"{gene}_") else f"{gene}_{unit}"


def normalise_measured_pairs(
    df: pd.DataFrame,
    collapse: str = "most_significant",
) -> pd.DataFrame:
    """Bring a measured CRISPRi pair table into one sorted, joinable layout.

    Accepts a pair_name layout ("GENE_enhA+enhB", interaction_term, p_value) or an
    enhA/enhB layout (interaction_beta, source_file).

    Args:
        df: Measured pair table.
        collapse: Repeated pairs: "most_significant" keeps the smallest p-value,
            "mean" averages coefficients, "none" keeps every row.

    Returns:
        Frame with gene, sorted fully qualified enhA and enhB, interaction_beta,
        measured_pval, source and n_records.

    Raises:
        ValueError: If neither layout is found.
    """
    if collapse not in {"most_significant", "mean", "none"}:
        raise ValueError(f"Unknown collapse policy '{collapse}'.")

    work = df.copy()
    work.columns = [str(c) for c in work.columns]

    if "pair_name" in work.columns:
        split = work["pair_name"].astype(str).str.split("+", n=1, expand=True)
        first = split[0].astype(str)
        match = first.str.extract(_QUALIFIED)
        gene = match["gene"]
        a_full = first
        b_full = [_qualify(g, u) for g, u in zip(gene, split[1].fillna(""))]
        value = pd.to_numeric(work.get("interaction_term"), errors="coerce")
        pval = pd.to_numeric(work.get("p_value"), errors="coerce")
        source = work.get("dataset", pd.Series("", index=work.index))

    elif {"enhA", "enhB"}.issubset(work.columns):
        a_full = work["enhA"].astype(str)
        b_full = work["enhB"].astype(str)
        gene = a_full.str.extract(_QUALIFIED)["gene"]
        value = pd.to_numeric(work.get("interaction_beta"), errors="coerce")
        # The adjusted table carries its p-value in a trailing unnamed column,
        # which pandas renames to "Unnamed: N" on read.
        def _looks_like_pval(column: str) -> bool:
            name = column.strip().lower()
            return (name == "" or name.startswith("unnamed")
                    or name in {"p_value", "pval", "measured_pval", "pvalue"})

        pcol = next((c for c in work.columns
                     if _looks_like_pval(c)
                     and pd.to_numeric(work[c], errors="coerce").notna().any()),
                    None)
        pval = (pd.to_numeric(work[pcol], errors="coerce") if pcol is not None
                else pd.Series(np.nan, index=work.index))
        source = work.get("source_file", pd.Series("", index=work.index))

    else:
        raise ValueError(
            "Cannot recognise this measured pair table: expected either a "
            "'pair_name' column or both 'enhA' and 'enhB'. Columns present: "
            f"{list(df.columns)}"
        )

    ordered = [tuple(sorted([str(a), str(b)])) for a, b in zip(a_full, b_full)]
    out = pd.DataFrame({
        "gene": gene.to_numpy(),
        "enhA": [p[0] for p in ordered],
        "enhB": [p[1] for p in ordered],
        "interaction_beta": value.to_numpy(),
        "measured_pval": pval.to_numpy(),
        "source": np.asarray(source, dtype=object),
    }).dropna(subset=["interaction_beta"])

    counts = out.groupby(["enhA", "enhB"]).size()
    n_duplicated = int((counts > 1).sum())

    if collapse == "none" or out.empty:
        out["n_records"] = out.groupby(["enhA", "enhB"])["enhA"].transform("size")
        return out.reset_index(drop=True)

    if n_duplicated:
        warnings.warn(
            f"{n_duplicated} enhancer pair(s) appear more than once in the "
            f"measured table; collapsing with policy '{collapse}'. Sample "
            f"sizes reported downstream are numbers of distinct pairs."
        )

    records = []
    for (a, b), group in out.groupby(["enhA", "enhB"], sort=False):
        ranked = group.sort_values("measured_pval", na_position="last")
        best = ranked.iloc[0]
        beta = (float(group["interaction_beta"].mean()) if collapse == "mean"
                else float(best["interaction_beta"]))
        records.append({
            "gene": best["gene"], "enhA": a, "enhB": b,
            "interaction_beta": beta,
            "measured_pval": float(best["measured_pval"])
            if pd.notna(best["measured_pval"]) else np.nan,
            "source": "|".join(sorted({str(s) for s in group["source"]})),
            "n_records": int(len(group)),
        })
    return pd.DataFrame(records)


def normalise_measured_singles(
    df: pd.DataFrame,
    enhancer_col: str = "enhancer",
    beta_col: str = "beta",
    pval_col: str | None = None,
    source_col: str = "source_dataset",
    collapse: str = "most_significant",
) -> pd.DataFrame:
    """Bring the measured single-enhancer table into one layout.

    Args:
        df: Measured single-enhancer coefficients.
        enhancer_col: Enhancer name column.
        beta_col: Coefficient column.
        pval_col: p-value column, if present.
        source_col: Source dataset column.
        collapse: As in normalise_measured_pairs; without p-values
            "most_significant" keeps the first record.

    Returns:
        Frame with enhancer, measured_beta, measured_pval, source and n_records.
    """
    work = df.copy()
    out = pd.DataFrame({
        "enhancer": work[enhancer_col].astype(str),
        "measured_beta": pd.to_numeric(work[beta_col], errors="coerce"),
        "measured_pval": (pd.to_numeric(work[pval_col], errors="coerce")
                          if pval_col and pval_col in work.columns
                          else np.nan),
        "source": (work[source_col].astype(str) if source_col in work.columns
                   else ""),
    }).dropna(subset=["measured_beta"])

    if collapse == "none" or out.empty:
        out["n_records"] = out.groupby("enhancer")["enhancer"].transform("size")
        return out.reset_index(drop=True)

    records = []
    for enhancer, group in out.groupby("enhancer", sort=False):
        ranked = group.sort_values("measured_pval", na_position="last")
        best = ranked.iloc[0]
        beta = (float(group["measured_beta"].mean()) if collapse == "mean"
                else float(best["measured_beta"]))
        records.append({
            "enhancer": enhancer, "measured_beta": beta,
            "measured_pval": float(best["measured_pval"])
            if pd.notna(best["measured_pval"]) else np.nan,
            "source": "|".join(sorted({str(s) for s in group["source"]})),
            "n_records": int(len(group)),
        })
    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# G1. Motif pairs against the enhancer pair, in silico and measured
# --------------------------------------------------------------------------- #

def link_motif_pairs_to_enhancers(
    pair_agg: pd.DataFrame,
    enh_inters: pd.DataFrame,
    measured_pairs: pd.DataFrame | None = None,
    enh_value_col: str = "coef",
    enh_a_col: str = "full_a",
    enh_b_col: str = "full_b",
    enh_qval_col: str = "qval",
) -> pd.DataFrame:
    """Join aggregated motif-pair interactions to the in silico and measured enhancer-pair coefficients.

    Args:
        pair_agg: Output of glm.aggregate_pairs_to_enhancer.
        enh_inters: Enhancer-level interactions after BH correction.
        measured_pairs: Output of normalise_measured_pairs.
        enh_value_col: Coefficient column of enh_inters.
        enh_a_col: First enhancer column of enh_inters.
        enh_b_col: Second enhancer column of enh_inters.
        enh_qval_col: Adjusted p-value column of enh_inters.

    Returns:
        One row per enhancer pair on the motif side with the motif aggregates,
        enh_coef, enh_qval and, if supplied, interaction_beta and measured_pval;
        unmatched values stay missing.
    """
    if pair_agg.empty:
        return pd.DataFrame()

    left = pair_agg.join(_pair_key(pair_agg, "enhA", "enhB"))

    right = enh_inters.join(_pair_key(enh_inters, enh_a_col, enh_b_col))
    keep = {"_key_a": "_key_a", "_key_b": "_key_b",
            enh_value_col: "enh_coef"}
    if enh_qval_col in right.columns:
        keep[enh_qval_col] = "enh_qval"
    right = right[list(keep)].rename(columns=keep)
    right = right.drop_duplicates(["_key_a", "_key_b"])

    merged = left.merge(right, on=["_key_a", "_key_b"], how="left")

    if measured_pairs is not None and not measured_pairs.empty:
        meas = measured_pairs.join(_pair_key(measured_pairs, "enhA", "enhB"))
        cols = ["_key_a", "_key_b", "interaction_beta", "measured_pval"]
        cols = [c for c in cols if c in meas.columns]
        merged = merged.merge(meas[cols].drop_duplicates(["_key_a", "_key_b"]),
                              on=["_key_a", "_key_b"], how="left")

    return merged.drop(columns=["_key_a", "_key_b"])


def summarise_pair_link(
    linked: pd.DataFrame,
    motif_cols: tuple[str, ...] = ("motif_mean", "motif_strongest",
                                   "motif_sum", "motif_max"),
    enh_col: str = "enh_coef",
    measured_col: str = "interaction_beta",
    enh_qval_col: str = "enh_qval",
    measured_p_col: str = "measured_pval",
    alpha: float = 0.05,
) -> dict:
    """Correlation and sign agreement of each motif aggregate with the in silico and measured enhancer coefficients.

    Args:
        linked: Output of link_motif_pairs_to_enhancers.
        motif_cols: Aggregates to report; motif_mean is the primary one.
        enh_col: In silico enhancer coefficient column.
        measured_col: Measured coefficient column.
        enh_qval_col: Enhancer-level adjusted p-value column.
        measured_p_col: Measured p-value column.
        alpha: Significance threshold for both assays.

    Returns:
        Nested dictionary by target and aggregate with correlations, sign
        agreement, leave-one-out range and the same restricted to significant pairs.
    """
    out: dict = {"alpha": alpha, "n_pairs": int(len(linked))}

    targets = [("insilico", enh_col, enh_qval_col)]
    if measured_col in linked.columns:
        targets.append(("measured", measured_col, measured_p_col))

    for target_name, target_col, p_col in targets:
        if target_col not in linked.columns:
            continue
        block: dict = {}
        available = linked.dropna(subset=[target_col])
        block["n_available"] = int(len(available))

        resolvable = available
        if p_col in available.columns:
            resolvable = available[available[p_col] < alpha]
        block["n_resolvable"] = int(len(resolvable))

        for col in motif_cols:
            if col not in linked.columns:
                continue
            entry = {}
            entry.update(_corr_stats(available[col], available[target_col]))
            entry.update(_sign_concordance(available[col],
                                           available[target_col]))
            entry.update(_jackknife_influence(available[col],
                                              available[target_col]))
            entry.update(_corr_stats(resolvable[col], resolvable[target_col],
                                     prefix="sig_"))
            block[col] = entry
        out[target_name] = block

    return out


# --------------------------------------------------------------------------- #
# G5. The same link with gene identity removed
# --------------------------------------------------------------------------- #

def summarise_within_gene_link(
    linked: pd.DataFrame,
    x_col: str = "motif_mean",
    y_col: str = "enh_coef",
    gene_col: str = "gene",
    min_pairs: int = 2,
) -> dict:
    """Split the motif-enhancer correlation into within-gene and between-gene parts.

    Args:
        linked: Output of link_motif_pairs_to_enhancers.
        x_col: Motif-side column.
        y_col: Enhancer-side column.
        gene_col: Gene column.
        min_pairs: Minimum pairs per gene for the within-gene view.

    Returns:
        Dictionary with pooled, within-gene (gene-centred) and between-gene
        (gene means) correlations, per-gene Spearman correlations and a sign test.
    """
    work = linked.dropna(subset=[x_col, y_col, gene_col]).copy()
    out: dict = {"x_col": x_col, "y_col": y_col, "min_pairs": min_pairs}

    out.update(_corr_stats(work[x_col], work[y_col], prefix="pooled_"))
    out.update(_jackknife_influence(work[x_col], work[y_col]))

    sizes = work.groupby(gene_col)[x_col].transform("size")
    kept = work[sizes >= min_pairs].copy()
    out["n_genes_total"] = int(work[gene_col].nunique())
    out["n_genes_kept"] = int(kept[gene_col].nunique())
    out["n_pairs_kept"] = int(len(kept))

    if not kept.empty:
        kept["_xc"] = kept[x_col] - kept.groupby(gene_col)[x_col].transform("mean")
        kept["_yc"] = kept[y_col] - kept.groupby(gene_col)[y_col].transform("mean")
        out.update(_corr_stats(kept["_xc"], kept["_yc"], prefix="within_"))
    else:
        out.update(_corr_stats([], [], prefix="within_"))

    means = work.groupby(gene_col)[[x_col, y_col]].mean()
    out.update(_corr_stats(means[x_col], means[y_col], prefix="between_"))

    # Per-gene rank correlations, which need at least three pairs to exist.
    per_gene = []
    for gene, group in work.groupby(gene_col):
        if len(group) < 3 or group[x_col].nunique() < 2 or group[y_col].nunique() < 2:
            continue
        rho = stats.spearmanr(group[x_col], group[y_col])[0]
        if np.isfinite(rho):
            per_gene.append({"gene": gene, "n_pairs": int(len(group)),
                             "spearman_rho": float(rho)})

    out["per_gene"] = pd.DataFrame(per_gene)
    if per_gene:
        rhos = np.array([g["spearman_rho"] for g in per_gene])
        positive = int((rhos > 0).sum())
        nonzero = int((rhos != 0).sum())
        out["per_gene_n"] = len(rhos)
        out["per_gene_median_rho"] = float(np.median(rhos))
        out["per_gene_n_positive"] = positive
        out["per_gene_sign_p"] = (float(stats.binomtest(positive, nonzero, 0.5).pvalue)
                                  if nonzero else np.nan)
    else:
        out["per_gene_n"] = 0
        out["per_gene_median_rho"] = np.nan
        out["per_gene_n_positive"] = 0
        out["per_gene_sign_p"] = np.nan

    return out


# --------------------------------------------------------------------------- #
# G2. Motif content against the enhancer's own knockout effect
# --------------------------------------------------------------------------- #

def aggregate_motifs_per_enhancer(
    motifs: pd.DataFrame,
    coop: pd.DataFrame | None = None,
    enhancer_col: str = "enhancer",
    value_col: str = "delta",
    qval_col: str = "qval",
    coop_col: str = "cooperativity",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Aggregate motif main effects to one row per enhancer.

    Args:
        motifs: build_motif_effects output after BH correction.
        coop: build_cooperativity_table output, joined on the enhancer.
        enhancer_col: Enhancer column.
        value_col: Motif effect column.
        qval_col: Adjusted p-value column.
        coop_col: Cooperativity column of coop.
        alpha: Significance threshold.

    Returns:
        One row per enhancer with motif_strongest (primary), motif_max_abs,
        motif_sum_abs, motif_mean, motif_median, n_motifs, n_significant and,
        with coop, cooperativity and n_seqlets.
    """
    if motifs.empty:
        return pd.DataFrame()

    work = motifs.dropna(subset=[value_col]).copy()
    work["_abs"] = work[value_col].abs()
    work["_sig"] = (work[qval_col] < alpha) if qval_col in work.columns else False

    agg = (work.groupby(enhancer_col)
               .agg(motif_strongest=(value_col, "min"),
                    motif_mean=(value_col, "mean"),
                    motif_median=(value_col, "median"),
                    motif_max_abs=("_abs", "max"),
                    motif_sum_abs=("_abs", "sum"),
                    n_motifs=(value_col, "size"),
                    n_significant=("_sig", "sum"))
               .reset_index()
               .rename(columns={enhancer_col: "enhancer"}))

    if coop is not None and not coop.empty:
        keep = [c for c in ["enhancer", coop_col, "n_seqlets", "qval"]
                if c in coop.columns]
        block = coop[keep].drop_duplicates("enhancer").rename(
            columns={coop_col: "cooperativity", "qval": "coop_qval"})
        agg = agg.merge(block, on="enhancer", how="left")

    return agg


def link_motif_effects_to_enhancers(
    enh_agg: pd.DataFrame,
    enh_mains: pd.DataFrame,
    measured_singles: pd.DataFrame | None = None,
    enh_value_col: str = "coef",
    enh_id_col: str = "full_a",
    enh_qval_col: str = "qval",
) -> pd.DataFrame:
    """Join per-enhancer motif aggregates to the enhancer's in silico and measured knockout effect.

    Args:
        enh_agg: Output of aggregate_motifs_per_enhancer.
        enh_mains: Enhancer-level main effects after BH correction.
        measured_singles: Output of normalise_measured_singles.
        enh_value_col: Coefficient column of enh_mains.
        enh_id_col: Fully qualified enhancer column of enh_mains.
        enh_qval_col: Adjusted p-value column of enh_mains.

    Returns:
        One row per enhancer on the motif side with enh_coef, enh_qval and, if
        supplied, measured_beta; warns how many enhancers did not match.
    """
    if enh_agg.empty:
        return pd.DataFrame()

    keep = {enh_id_col: "enhancer", enh_value_col: "enh_coef"}
    if enh_qval_col in enh_mains.columns:
        keep[enh_qval_col] = "enh_qval"
    if "gene" in enh_mains.columns:
        keep["gene"] = "gene"
    right = enh_mains[list(keep)].rename(columns=keep).drop_duplicates("enhancer")

    merged = enh_agg.merge(right, on="enhancer", how="left")

    unmatched = int(merged["enh_coef"].isna().sum())
    if unmatched:
        warnings.warn(
            f"{unmatched} of {len(merged)} enhancer(s) on the motif side have "
            f"no enhancer-level coefficient. Check for compound identifiers "
            f"that name two genes."
        )

    if measured_singles is not None and not measured_singles.empty:
        cols = [c for c in ["enhancer", "measured_beta", "measured_pval"]
                if c in measured_singles.columns]
        merged = merged.merge(measured_singles[cols].drop_duplicates("enhancer"),
                              on="enhancer", how="left")

    return merged


def summarise_motif_enhancer_link(
    linked: pd.DataFrame,
    motif_cols: tuple[str, ...] = ("motif_strongest", "motif_sum_abs",
                                   "motif_mean", "cooperativity"),
    enh_col: str = "enh_coef",
    measured_col: str = "measured_beta",
    alpha: float = 0.05,
    enh_qval_col: str = "enh_qval",
) -> dict:
    """Correlation of each motif aggregate with the in silico and measured enhancer main effects.

    Args:
        linked: Output of link_motif_effects_to_enhancers.
        motif_cols: Aggregates to report; motif_strongest is the primary one.
        enh_col: In silico main effect column.
        measured_col: Measured main effect column.
        alpha: Significance threshold for enh_qval_col.
        enh_qval_col: Enhancer-level adjusted p-value column.

    Returns:
        Nested dictionary by target and aggregate; sign agreement only for
        aggregates that can take either sign.
    """
    out: dict = {"alpha": alpha, "n_enhancers": int(len(linked))}
    signed = {"motif_strongest", "motif_mean", "motif_median", "cooperativity"}

    targets = [("insilico", enh_col), ("measured", measured_col)]
    for target_name, target_col in targets:
        if target_col not in linked.columns:
            continue
        available = linked.dropna(subset=[target_col])
        block: dict = {"n_available": int(len(available))}

        resolvable = available
        if enh_qval_col in available.columns:
            resolvable = available[available[enh_qval_col] < alpha]
        block["n_resolvable"] = int(len(resolvable))

        for col in motif_cols:
            if col not in linked.columns:
                continue
            entry = {}
            entry.update(_corr_stats(available[col], available[target_col]))
            if col in signed:
                entry.update(_sign_concordance(available[col],
                                               available[target_col]))
            entry.update(_corr_stats(resolvable[col], resolvable[target_col],
                                     prefix="sig_"))
            block[col] = entry
        out[target_name] = block

    return out


# --------------------------------------------------------------------------- #
# G3. Does the spacing effect survive conditioning on effect size
# --------------------------------------------------------------------------- #

def summarise_spacing_partial(
    pairs: pd.DataFrame,
    spacing_col: str = "spacing_bp",
    value_col: str = "interaction",
    delta_a_col: str = "delta_a",
    delta_b_col: str = "delta_b",
    log_spacing: bool = True,
) -> dict:
    """Test whether the interaction-spacing correlation survives removing the summed main effects.

    Args:
        pairs: build_pairwise_interactions output with spacing attached.
        spacing_col: Edge-to-edge gap column (bp).
        value_col: Interaction column.
        delta_a_col: First main effect column.
        delta_b_col: Second main effect column.
        log_spacing: Also report the statistics on log10(spacing + 1).

    Returns:
        Dictionary of raw, confound and partial correlations (raw and rank scale),
        each with its sample size.

    Raises:
        KeyError: If spacing_col is missing.
    """
    if spacing_col not in pairs.columns:
        raise KeyError(
            f"No '{spacing_col}' column. Pass the seqlet table to "
            f"build_pairwise_interactions so coordinates are attached."
        )

    needed = [spacing_col, value_col, delta_a_col, delta_b_col]
    work = pairs.dropna(subset=needed).copy()
    work["_additive"] = work[delta_a_col] + work[delta_b_col]

    out: dict = {"n": int(len(work))}
    out.update(_corr_stats(work[spacing_col], work[value_col], prefix="raw_"))
    out.update(_corr_stats(work[spacing_col], work["_additive"],
                           prefix="confound_"))
    out.update(_corr_stats(work["_additive"], work[value_col],
                           prefix="saturation_"))

    def partials(spacing: pd.Series, tag: str) -> None:
        r, p, n = partial_correlation(spacing, work[value_col], work["_additive"])
        out[f"{tag}partial_r"], out[f"{tag}partial_p"] = r, p
        out[f"{tag}partial_n"] = n

        # Rank-transform every variable before partialling, which gives the
        # rank analogue and is the appropriate scale here because the
        # interaction magnitudes are uncalibrated.
        r_s, p_s, n_s = partial_correlation(
            stats.rankdata(spacing), stats.rankdata(work[value_col]),
            stats.rankdata(work["_additive"]))
        out[f"{tag}partial_rho"], out[f"{tag}partial_rho_p"] = r_s, p_s
        out[f"{tag}partial_rho_n"] = n_s

        # The reverse: does the saturation coupling survive removing spacing.
        r_rev, p_rev, n_rev = partial_correlation(
            work["_additive"], work[value_col], spacing)
        out[f"{tag}reverse_partial_r"] = r_rev
        out[f"{tag}reverse_partial_p"] = p_rev
        out[f"{tag}reverse_partial_n"] = n_rev

    partials(work[spacing_col], "")
    if log_spacing:
        work["_log_spacing"] = np.log10(work[spacing_col].clip(lower=0) + 1)
        out.update(_corr_stats(work["_log_spacing"], work[value_col],
                               prefix="log_raw_"))
        partials(work["_log_spacing"], "log_")

    return out


# --------------------------------------------------------------------------- #
# Text summary, in the format used by report.py
# --------------------------------------------------------------------------- #

def _p(value: float | None) -> str:
    """Format a p-value compactly, as in report.py."""
    if value is None or not np.isfinite(value):
        return "p=n/a"
    return f"p={value:.1e}" if value < 1e-4 else f"p={value:.4f}"


def _corr_line(entry: dict, label: str, prefix: str = "") -> str:
    """Format one correlation block from _corr_stats as a line."""
    n = entry.get(f"{prefix}n", 0)
    r = entry.get(f"{prefix}pearson_r", np.nan)
    rho = entry.get(f"{prefix}spearman_rho", np.nan)
    if not np.isfinite(r if r is not None else np.nan):
        return f"  {label:<34} not estimable (n={n})"
    return (f"  {label:<34} n={n:<5} r={r:+.3f} "
            f"{_p(entry.get(f'{prefix}pearson_p')):<12} rho={rho:+.3f} "
            f"{_p(entry.get(f'{prefix}spearman_p'))}")


def format_cross_level_summary(
    pair_link: dict | None = None,
    within_gene: dict | None = None,
    motif_link: dict | None = None,
    spacing: dict | None = None,
) -> str:
    """Render the cross-level analyses as a text block in the summary format.

    Args:
        pair_link: Output of summarise_pair_link.
        within_gene: Output of summarise_within_gene_link.
        motif_link: Output of summarise_motif_enhancer_link.
        spacing: Output of summarise_spacing_partial.

    Returns:
        The text block.
    """
    rule, thin = "=" * 78, "-" * 78
    lines = [rule, "CROSS-LEVEL LINKS", rule, ""]

    lines += [thin, "G1.  MOTIF PAIRS AGAINST THE ENHANCER PAIR", thin]
    if not pair_link:
        lines.append("  [not supplied]")
    else:
        lines.append(f"  enhancer pairs on the motif side: {pair_link['n_pairs']}")
        for target in ("insilico", "measured"):
            block = pair_link.get(target)
            if not block:
                continue
            name = ("in silico enhancer beta_ij" if target == "insilico"
                    else "measured CRISPRi beta_ij")
            lines.append(f"  vs {name}  (matched {block['n_available']}, "
                         f"resolvable {block['n_resolvable']})")
            for col, entry in block.items():
                if not isinstance(entry, dict):
                    continue
                lines.append(_corr_line(entry, f"    {col}"))
                if np.isfinite(entry.get("concordance", np.nan)):
                    lines.append(
                        f"    {'sign agreement':<32} "
                        f"{entry['n_concordant']}/{entry['n_sign']} = "
                        f"{100 * entry['concordance']:.1f}%  "
                        f"[{100 * entry['ci_lo']:.1f}%,{100 * entry['ci_hi']:.1f}%]"
                        f"  {_p(entry['binom_p'])}")
                if np.isfinite(entry.get("jack_max_shift", np.nan)):
                    lines.append(
                        f"    {'leave-one-out r range':<32} "
                        f"[{entry['jack_r_min']:+.3f},{entry['jack_r_max']:+.3f}]"
                        f"  max shift {entry['jack_max_shift']:.3f}")
                lines.append(_corr_line(entry, "    restricted to significant",
                                        prefix="sig_"))
    lines.append("")

    lines += [thin, "G5.  THE SAME LINK WITH GENE IDENTITY REMOVED", thin]
    if not within_gene:
        lines.append("  [not supplied]")
    else:
        lines.append(f"  {within_gene['x_col']} against {within_gene['y_col']}")
        lines.append(_corr_line(within_gene, "pooled", prefix="pooled_"))
        lines.append(_corr_line(within_gene, "within gene, centred",
                                prefix="within_"))
        lines.append(_corr_line(within_gene, "between gene, per-gene means",
                                prefix="between_"))
        lines.append(f"  {'genes kept':<34} {within_gene['n_genes_kept']} of "
                     f"{within_gene['n_genes_total']}, "
                     f"{within_gene['n_pairs_kept']} pairs")
        if within_gene.get("per_gene_n"):
            lines.append(
                f"  {'per-gene rho':<34} n={within_gene['per_gene_n']:<5} "
                f"median={within_gene['per_gene_median_rho']:+.3f}  "
                f"positive={within_gene['per_gene_n_positive']}  "
                f"{_p(within_gene['per_gene_sign_p'])}")
        if np.isfinite(within_gene.get("jack_max_shift", np.nan)):
            lines.append(
                f"  {'leave-one-out r range':<34} "
                f"[{within_gene['jack_r_min']:+.3f},"
                f"{within_gene['jack_r_max']:+.3f}]")
    lines.append("")

    lines += [thin, "G2.  MOTIF CONTENT AGAINST THE ENHANCER MAIN EFFECT", thin]
    if not motif_link:
        lines.append("  [not supplied]")
    else:
        lines.append(f"  enhancers on the motif side: {motif_link['n_enhancers']}")
        for target in ("insilico", "measured"):
            block = motif_link.get(target)
            if not block:
                continue
            name = ("in silico enhancer beta_i" if target == "insilico"
                    else "measured CRISPRi beta_i")
            lines.append(f"  vs {name}  (matched {block['n_available']})")
            for col, entry in block.items():
                if not isinstance(entry, dict):
                    continue
                lines.append(_corr_line(entry, f"    {col}"))
                if np.isfinite(entry.get("concordance", np.nan)):
                    lines.append(
                        f"    {'sign agreement':<32} "
                        f"{entry['n_concordant']}/{entry['n_sign']} = "
                        f"{100 * entry['concordance']:.1f}%  "
                        f"{_p(entry['binom_p'])}")
    lines.append("")

    lines += [thin, "G3.  SPACING CONDITIONED ON THE SUMMED MAIN EFFECTS", thin]
    if not spacing:
        lines.append("  [not supplied]")
    else:
        lines.append(_corr_line(spacing, "spacing vs I, raw", prefix="raw_"))
        lines.append(_corr_line(spacing, "spacing vs delta_A+delta_B",
                                prefix="confound_"))
        lines.append(_corr_line(spacing, "delta_A+delta_B vs I",
                                prefix="saturation_"))
        lines.append(
            f"  {'spacing vs I | additive':<34} "
            f"n={spacing.get('partial_n', 0):<5} "
            f"partial r={spacing.get('partial_r', np.nan):+.3f} "
            f"{_p(spacing.get('partial_p'))}")
        lines.append(
            f"  {'same, on ranks':<34} "
            f"n={spacing.get('partial_rho_n', 0):<5} "
            f"partial rho={spacing.get('partial_rho', np.nan):+.3f} "
            f"{_p(spacing.get('partial_rho_p'))}")
        lines.append(
            f"  {'additive vs I | spacing':<34} "
            f"n={spacing.get('reverse_partial_n', 0):<5} "
            f"partial r={spacing.get('reverse_partial_r', np.nan):+.3f} "
            f"{_p(spacing.get('reverse_partial_p'))}")
        if "log_partial_r" in spacing:
            lines.append(_corr_line(spacing, "log10 spacing vs I, raw",
                                    prefix="log_raw_"))
            lines.append(
                f"  {'log10 spacing vs I | additive':<34} "
                f"n={spacing.get('log_partial_n', 0):<5} "
                f"partial r={spacing.get('log_partial_r', np.nan):+.3f} "
                f"{_p(spacing.get('log_partial_p'))}")
    lines += ["", rule, "END", rule]
    return "\n".join(lines)