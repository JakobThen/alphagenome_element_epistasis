"""Plain-text summary of the within-enhancer TF and enhancer-level tables.

Each section reports distributions, sample sizes, p-values and top-N extremes;
missing inputs are listed as missing. Entry points: build_summary, write_summary.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

_RULE = "=" * 78
_THIN = "-" * 78


# --------------------------------------------------------------------------- #
# Formatting primitives
# --------------------------------------------------------------------------- #

def _fmt_p(p: float | None) -> str:
    """Format a p-value compactly."""
    if p is None or not np.isfinite(p):
        return "p=n/a"
    if p < 1e-4:
        return f"p={p:.1e}"
    return f"p={p:.4f}"


def _describe(series: pd.Series, name: str, unit: str = "") -> str:
    """Summarise a distribution in one line: n, median, IQR and range."""
    values = pd.to_numeric(pd.Series(series), errors="coerce").dropna()
    if values.empty:
        return f"  {name:<34} no finite values"
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    return (f"  {name:<34} n={len(values):<5} "
            f"median={med:+.3f}  IQR=[{q1:+.3f},{q3:+.3f}]  "
            f"range=[{values.min():+.3f},{values.max():+.3f}]{unit}")


def _corr(x: pd.Series, y: pd.Series, label: str) -> str:
    """Pearson and Spearman correlation on the shared non-missing values, as one line."""
    frame = pd.DataFrame(dict(x=pd.to_numeric(pd.Series(x).reset_index(drop=True),
                                              errors="coerce"),
                              y=pd.to_numeric(pd.Series(y).reset_index(drop=True),
                                              errors="coerce"))).dropna()
    if len(frame) < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return f"  {label:<34} not estimable (n={len(frame)})"
    r, pr = stats.pearsonr(frame["x"], frame["y"])
    rho, prho = stats.spearmanr(frame["x"], frame["y"])
    return (f"  {label:<34} n={len(frame):<5} "
            f"r={r:+.3f} {_fmt_p(pr):<12} rho={rho:+.3f} {_fmt_p(prho)}")


def _direction_split(values: pd.Series, label: str,
                     alpha_flags: pd.Series | None = None) -> str:
    """Count negative and positive values, optionally with significant counts per sign."""
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if values.empty:
        return f"  {label:<34} no finite values"
    neg, pos = int((values < 0).sum()), int((values > 0).sum())
    extra = ""
    if alpha_flags is not None:
        flags = pd.Series(alpha_flags).reindex(values.index).fillna(False)
        extra = (f"  |  significant: {int((flags & (values < 0)).sum())} super, "
                 f"{int((flags & (values > 0)).sum())} sub")
    return (f"  {label:<34} super-additive(<0)={neg}  sub-additive(>0)={pos}"
            f"{extra}")


def _top(df: pd.DataFrame, value_col: str, label_cols: list[str],
         n: int, both_directions: bool = True,
         extra_cols: list[str] | None = None) -> list[str]:
    """List the named extremes of a column, taken from each direction separately."""
    if df is None or df.empty or value_col not in df.columns:
        return ["    (unavailable)"]

    work = df.dropna(subset=[value_col]).copy()
    if work.empty:
        return ["    (no finite values)"]

    label_cols = [c for c in label_cols if c in work.columns]
    extra_cols = [c for c in (extra_cols or []) if c in work.columns]

    def render(subset: pd.DataFrame, heading: str) -> list[str]:
        lines = [f"    {heading}"]
        if subset.empty:
            lines.append("      (none)")
            return lines
        for _, row in subset.iterrows():
            name = " / ".join(str(row[c]) for c in label_cols) or "?"
            tail = "".join(
                f"  {c}={row[c]:.3g}" if isinstance(row[c], (int, float, np.number))
                else f"  {c}={row[c]}" for c in extra_cols
            )
            lines.append(f"      {name:<44} {row[value_col]:+.4f}{tail}")
        return lines

    if both_directions:
        negative = work[work[value_col] < 0]
        positive = work[work[value_col] > 0]
        out = render(negative.nsmallest(min(n, len(negative)), value_col),
                     f"most negative (showing {min(n, len(negative))} "
                     f"of {len(negative)}):")
        out += render(positive.nlargest(min(n, len(positive)), value_col),
                      f"most positive (showing {min(n, len(positive))} "
                      f"of {len(positive)}):")
        return out
    return render(work.reindex(work[value_col].abs()
                               .sort_values(ascending=False).index).head(n),
                  f"largest by magnitude (top {n}):")


def _fdr_line(df: pd.DataFrame, label: str, qcol: str = "qval",
              alpha: float = 0.05) -> str:
    """Count the tests passing the FDR threshold."""
    if df is None or df.empty:
        return f"  {label:<34} not supplied"
    if qcol not in df.columns:
        return f"  {label:<34} n={len(df)}, no '{qcol}' column (uncorrected)"
    valid = df[qcol].notna()
    return (f"  {label:<34} {int((df.loc[valid, qcol] < alpha).sum())} of "
            f"{int(valid.sum())} at FDR<{alpha:g} "
            f"({int((~valid).sum())} untested)")


def _missing(name: str) -> list[str]:
    """Placeholder line for an input that was not supplied."""
    return [f"  [not supplied: {name}]"]


# --------------------------------------------------------------------------- #
# Section builders
# --------------------------------------------------------------------------- #

def _section_tf_motifs(motifs: pd.DataFrame | None, controls: pd.DataFrame | None,
                       alpha: float, top_n: int) -> list[str]:
    """Section 3.1: TF motif main effects and the control comparison."""
    out = ["", _THIN, "3.1  TF MOTIF MAIN EFFECTS (delta_i, log scale)", _THIN]
    if motifs is None or motifs.empty:
        return out + _missing("motifs")

    out.append(f"  motifs={len(motifs)}  enhancers={motifs['enhancer'].nunique()}"
               f"  distinct factors="
               f"{motifs['motif'].astype(str).str.split('#').str[0].nunique()}")
    out.append(_describe(motifs["delta"], "delta_i"))
    out.append(_fdr_line(motifs, "significant motifs", alpha=alpha))

    if "attribution" in motifs.columns:
        out.append(_corr(motifs["attribution"], motifs["delta"],
                         "attribution vs delta (supp A)"))
    else:
        out.append("  attribution vs delta               attribution not attached")

    tf = motifs.copy()
    tf["factor"] = tf["motif"].astype(str).str.split("#").str[0]
    per_tf = (tf.groupby("factor")
                .agg(n=("delta", "size"), mean=("delta", "mean"),
                     med=("delta", "median"))
                .sort_values("mean"))
    out.append("")
    out.append("  per factor (sorted by mean effect, strongest loss first):")
    out.append(f"      {'factor':<16}{'n':>5}{'mean':>10}{'median':>10}")
    for factor, row in per_tf.iterrows():
        out.append(f"      {factor:<16}{int(row['n']):>5}"
                   f"{row['mean']:>+10.3f}{row['med']:>+10.3f}")

    out.append("")
    out.append("  strongest individual motif effects:")
    out += _top(motifs, "delta", ["enhancer", "motif"], top_n,
                both_directions=False, extra_cols=["qval"])

    out += ["", "  SPECIFICITY CONTROL (supp B):"]
    if controls is None or controls.empty:
        out += ["    [not supplied: controls] - motif specificity is UNTESTED"]
    else:
        c, m = controls["delta"].dropna(), motifs["delta"].dropna()
        threshold = np.percentile(c, 5)
        u = stats.mannwhitneyu(m, c, alternative="less")
        out.append(f"    control windows={len(c)}  motifs={len(m)}")
        out.append(f"    control median={c.median():+.4f}   "
                   f"motif median={m.median():+.4f}")
        out.append(f"    control 5th percentile={threshold:+.4f}")
        out.append(f"    motifs below that threshold: "
                   f"{int((m < threshold).sum())}/{len(m)} "
                   f"({(m < threshold).mean():.1%})")
        out.append(f"    Mann-Whitney (motifs < controls): {_fmt_p(u.pvalue)}")
    return out


def _section_tf_pairs(pairs: pd.DataFrame | None, alpha: float,
                      top_n: int) -> list[str]:
    """Section 3.2: pairwise TF motif interactions."""
    out = ["", _THIN, "3.2  TF MOTIF PAIRWISE INTERACTIONS (I, log scale)", _THIN]
    if pairs is None or pairs.empty:
        return out + _missing("pairs")

    out.append(f"  pairs={len(pairs)}  enhancers={pairs['enhancer'].nunique()}")
    out.append(_describe(pairs["interaction"], "I"))
    out.append(_direction_split(pairs["interaction"], "direction",
                                pairs.get("significant")))
    out.append(_fdr_line(pairs, "significant pairs", alpha=alpha))

    if "spacing_bp" in pairs.columns:
        out.append(_corr(pairs["spacing_bp"], pairs["interaction"],
                         "spacing vs I (supp D)"))
        out.append(_describe(pairs["spacing_bp"], "spacing (bp)"))
    else:
        out.append("  spacing vs I                       spacing not attached")

    out += ["", "  extreme pairs:"]
    out += _top(pairs, "interaction", ["enhancer", "motif_a", "motif_b"],
                top_n, extra_cols=["qval", "spacing_bp"])

    # Recurrent grammar: the same factor pair seen in several enhancers.
    grammar = pairs.copy()
    grammar["fa"] = grammar["motif_a"].astype(str).str.split("#").str[0]
    grammar["fb"] = grammar["motif_b"].astype(str).str.split("#").str[0]
    ordered = grammar.apply(lambda r: tuple(sorted([r["fa"], r["fb"]])), axis=1)
    grammar["pair"] = [f"{a} x {b}" for a, b in ordered]
    agg = (grammar.groupby("pair")["interaction"]
                  .agg(n="size", mean="mean", median="median")
                  .query("n >= 2").sort_values("mean"))
    out += ["", f"  recurrent factor pairs (seen in 2+ enhancers, n={len(agg)}):"]
    if agg.empty:
        out.append("      (none recur)")
    else:
        out.append(f"      {'factor pair':<28}{'n':>4}{'mean I':>10}{'median':>10}")
        for pair, row in agg.iterrows():
            out.append(f"      {pair:<28}{int(row['n']):>4}"
                       f"{row['mean']:>+10.3f}{row['median']:>+10.3f}")
    return out


def _section_tf_coop(coop_atac: pd.DataFrame | None, coop_rna: pd.DataFrame | None,
                     alpha: float, top_n: int) -> list[str]:
    """Section 3.3: TF cooperativity per enhancer on ATAC and RNA."""
    out = ["", _THIN, "3.3  TF COOPERATIVITY PER ENHANCER", _THIN]
    if coop_atac is None or coop_atac.empty:
        return out + _missing("coop_atac")

    out.append(f"  enhancers={len(coop_atac)}")
    out.append(_describe(coop_atac["cooperativity"], "I (ATAC)"))
    out.append(_direction_split(coop_atac["cooperativity"], "direction",
                                coop_atac.get("significant")))
    out.append(_fdr_line(coop_atac, "significant enhancers", alpha=alpha))
    out.append(_corr(coop_atac["additive_prediction"], coop_atac["observed"],
                     "additive vs observed (main B)"))

    if "n_seqlets" in coop_atac.columns:
        out += ["", "  by motif count (supp C):"]
        out.append(f"      {'n motifs':<10}{'n enh':>7}{'median I':>12}"
                   f"{'mean I':>10}{'test vs 0':>18}")
        for n_motifs, group in coop_atac.groupby("n_seqlets"):
            values = group["cooperativity"].dropna()
            if len(values) >= 3:
                test = (stats.wilcoxon(values) if len(values) >= 10
                        else stats.ttest_1samp(values, 0.0))
                verdict = _fmt_p(test.pvalue)
            else:
                verdict = "n<3"
            out.append(f"      {int(n_motifs):<10}{len(values):>7}"
                       f"{values.median():>+12.3f}{values.mean():>+10.3f}"
                       f"{verdict:>18}")

    out += ["", "  extreme enhancers:"]
    out += _top(coop_atac, "cooperativity", ["enhancer"], top_n,
                extra_cols=["n_seqlets", "qval"])

    out += ["", "  ATAC vs RNA (main F):"]
    if coop_rna is None or coop_rna.empty:
        out += ["    [not supplied: coop_rna]"]
    else:
        merged = coop_atac.merge(coop_rna, on="enhancer",
                                 suffixes=("_atac", "_rna"))
        if merged.empty:
            out.append("    no shared enhancers")
        else:
            concordant = (np.sign(merged["cooperativity_atac"])
                          == np.sign(merged["cooperativity_rna"]))
            out.append(_corr(merged["cooperativity_atac"],
                             merged["cooperativity_rna"], "ATAC vs RNA coop"))
            out.append(f"    sign concordant: {int(concordant.sum())}/"
                       f"{len(merged)} ({concordant.mean():.1%})")
    return out


def _section_tf_artefacts(pairs: pd.DataFrame | None, split: pd.DataFrame | None,
                          floor_df: pd.DataFrame | None,
                          dose: pd.DataFrame | None) -> list[str]:
    """Section 3.4: saturation, floor and dose-response diagnostics."""
    out = ["", _THIN, "3.4  TF ARTEFACT DIAGNOSTICS", _THIN]

    out.append("  saturation, naive (supp E, mathematically coupled):")
    if pairs is not None and not pairs.empty and \
            {"delta_a", "delta_b"}.issubset(pairs.columns):
        out.append(_corr(pairs["delta_a"] + pairs["delta_b"],
                         pairs["interaction"], "  I vs delta_A+delta_B"))
    else:
        out.append("    main effects not attached")

    if pairs is not None and "baseline" in pairs.columns:
        out.append(_corr(pairs["baseline"], pairs["interaction"],
                         "  I vs baseline y0 (supp G)"))

    out += ["", "  saturation, decoupled (supp F):"]
    if split is None or split.empty:
        out += ["    [not supplied: split] - coupling is UNTESTED"]
    else:
        out.append(_corr(split["additive_split"], split["interaction_split"],
                         "  split-half I vs additive"))
        if {"additive_full", "interaction_full"}.issubset(split.columns):
            out.append(_corr(split["additive_full"], split["interaction_full"],
                             "  full-data (coupled)"))
        if "log_baseline" in split.columns:
            try:
                from element_epistasis.tf_cooperativity import partial_correlation
                r, p, n = partial_correlation(split["additive_split"],
                                              split["interaction_split"],
                                              split["log_baseline"])
                out.append(f"    {'partial r | log y0':<34} n={n:<5} "
                           f"r={r:+.3f} {_fmt_p(p)}")
            except Exception as error:
                out.append(f"    partial correlation failed: {error}")

    out += ["", "  floor check (supp H):"]
    if floor_df is None or floor_df.empty:
        out += ["    [not supplied: floor_df]"]
    else:
        below = floor_df["null_below_floor"]
        out.append(f"    empirical floor = {floor_df['floor'].iloc[0]:.4g}")
        out.append(f"    pairs whose null falls below it: {int(below.sum())}/"
                   f"{len(floor_df)} ({below.mean():.1%})")
        if "headroom" in floor_df.columns:
            out.append(_describe(floor_df["headroom"],
                                 "  headroom (log units above floor)"))

    out += ["", "  dose response across orders (supp I):"]
    if dose is None or dose.empty:
        out += ["    [not supplied: dose]"]
    else:
        out.append(f"      {'order':<8}{'n':>7}{'mean dev':>12}"
                   f"{'median dev':>12}{'mean pred':>12}")
        for order, group in dose.groupby("n_scrambled"):
            out.append(f"      {int(order):<8}{len(group):>7}"
                       f"{group['deviation'].mean():>+12.3f}"
                       f"{group['deviation'].median():>+12.3f}"
                       f"{group['predicted'].mean():>+12.3f}")
        if "scrambled_bp" in dose.columns and dose["scrambled_bp"].notna().any():
            out.append(_corr(dose["scrambled_bp"], dose["deviation"],
                             "  scrambled bp vs deviation"))
    return out


def _section_enhancer(enh_mains: pd.DataFrame | None, enh_inters: pd.DataFrame | None,
                      atac_inters: pd.DataFrame | None, enh_coop: pd.DataFrame | None,
                      single_comp: pd.DataFrame | None, pair_comp: pd.DataFrame | None,
                      interdependency: pd.DataFrame | None,
                      comparison: pd.DataFrame | None,
                      validation_summaries: dict | None,
                      trans_by_gene: pd.DataFrame | None,
                      trans_by_locus: pd.DataFrame | None,
                      coop_atac_gene: pd.DataFrame | None,
                      coop_atac_locus: pd.DataFrame | None,
                      alpha: float, top_n: int) -> list[str]:
    """Section 4: enhancer-level GLM effects, validation, interdependency and model comparison."""
    out = ["", _RULE, "4.  ENHANCER LEVEL (GLM, log link)", _RULE]

    out += ["", "4.1  MAIN EFFECTS (beta_i)"]
    if enh_mains is None or enh_mains.empty:
        out += _missing("enh_mains")
    else:
        out.append(f"  terms={len(enh_mains)}  genes={enh_mains['gene'].nunique()}")
        out.append(_describe(enh_mains["coef"], "beta_i"))
        out.append(_fdr_line(enh_mains, "significant", alpha=alpha))
        out += _top(enh_mains, "coef", ["full_a"], top_n,
                    both_directions=False, extra_cols=["qval"])

    out += ["", "4.2  PAIRWISE INTERACTIONS (beta_ij)"]
    if enh_inters is None or enh_inters.empty:
        out += _missing("enh_inters")
    else:
        out.append(f"  pairs={len(enh_inters)}  "
                   f"genes={enh_inters['gene'].nunique()}")
        out.append(_describe(enh_inters["coef"], "beta_ij"))
        out.append(_direction_split(enh_inters["coef"], "direction",
                                    enh_inters.get("significant")))
        out.append(_fdr_line(enh_inters, "significant pairs", alpha=alpha))
        if "distance_bp" in enh_inters.columns:
            out.append(_corr(enh_inters["distance_bp"], enh_inters["coef"],
                             "distance vs beta_ij"))
            out.append(_describe(enh_inters["distance_bp"], "distance (bp)"))
        if {"main_a", "main_b"}.issubset(enh_inters.columns):
            out.append(_corr(enh_inters["main_a"] + enh_inters["main_b"],
                             enh_inters["coef"], "saturation (additive)"))
        if "baseline" in enh_inters.columns:
            out.append(_corr(enh_inters["baseline"], enh_inters["coef"],
                             "saturation (baseline)"))
        out += ["", "  extreme pairs:"]
        out += _top(enh_inters, "coef", ["gene", "term"], top_n,
                    extra_cols=["qval", "distance_bp"])

    out += ["", "4.3  ATAC VS RNA INTERACTIONS"]
    if atac_inters is None or atac_inters.empty or enh_inters is None:
        out += _missing("atac_inters")
    else:
        merged = enh_inters.merge(atac_inters, on=["gene", "term"],
                                  suffixes=("_rna", "_atac"))
        if merged.empty:
            out.append("  no shared terms")
        else:
            concordant = (np.sign(merged["coef_rna"])
                          == np.sign(merged["coef_atac"]))
            out.append(_corr(merged["coef_atac"], merged["coef_rna"],
                             "ATAC vs RNA beta_ij"))
            out.append(f"  sign concordant: {int(concordant.sum())}/{len(merged)}"
                       f" ({concordant.mean():.1%})")

    out += ["", "4.4  ADDITIVITY PER GENE"]
    if enh_coop is None or enh_coop.empty:
        out += _missing("enh_coop")
    else:
        value_col = ("cooperativity" if "cooperativity" in enh_coop.columns
                     else None)
        if value_col:
            out.append(_describe(enh_coop[value_col], "I per gene"))
            out.append(_direction_split(enh_coop[value_col], "direction"))
            out += _top(enh_coop, value_col, ["name"], top_n)

    out += ["", "4.5  VALIDATION AGAINST CRISPRi"]
    validation_summaries = validation_summaries or {}
    if not validation_summaries:
        out += _missing("validation_summaries")
    else:
        for label, summary in validation_summaries.items():
            if not summary or not summary.get("n"):
                continue
            out.append(f"  {label}:")
            out.append(f"    Pearson r={summary.get('pearson_r', float('nan')):+.3f} "
                       f"{_fmt_p(summary.get('pearson_p'))}   "
                       f"Spearman rho={summary.get('spearman_rho', float('nan')):+.3f} "
                       f"{_fmt_p(summary.get('spearman_p'))}   "
                       f"slope={summary.get('slope', float('nan')):+.3f}")
            out.append(f"    {'subset':<24}{'concordant':>12}{'rate':>9}"
                       f"{'95% CI':>18}{'vs chance':>12}")
            for suffix, name in [("", "all measured"),
                                 ("_insilico", "in silico significant")]:
                n = summary.get(f"n{suffix}", 0)
                if not n:
                    out.append(f"    {name:<24}{'0':>12}{'-':>9}{'-':>18}{'-':>12}")
                    continue
                rate = summary.get(f"concordance{suffix}", float("nan"))
                lo = summary.get(f"ci_lo{suffix}", float("nan"))
                hi = summary.get(f"ci_hi{suffix}", float("nan"))
                out.append(
                    f"    {name:<24}"
                    f"{summary.get(f'n_concordant{suffix}', 0)}/{n:<10}"
                    f"{rate:>8.1%}"
                    f"{f'[{lo:.1%},{hi:.1%}]':>18}"
                    f"{_fmt_p(summary.get(f'binom_p{suffix}')):>12}")
            out.append("    (an interval spanning 50% carries no evidence of "
                       "agreement either way)")

    out += ["", "4.6  INTERDEPENDENCY (cis vs trans)"]
    if interdependency is None or interdependency.empty:
        out += _missing("interdependency")
    else:
        cis = interdependency[interdependency["is_cis"]]
        trans = interdependency[~interdependency["is_cis"]]
        out.append(f"  cells={len(interdependency)}  "
                   f"cis={len(cis)}  trans={len(trans)}")
        out.append(_describe(cis["delta"], "cis delta"))
        out.append(_describe(trans["delta"], "trans delta"))
        if "qval" in interdependency.columns:
            out.append(f"  significant trans effects: "
                       f"{int((trans['qval'] < alpha).sum())}/{len(trans)}")
        if len(trans):
            per_gene = trans.groupby("gene")["delta"].agg(["size", "mean"])
            per_gene = per_gene.sort_values("mean")
            out.append("  strongest mean trans effect by gene:")
            for gene, row in per_gene.head(top_n).iterrows():
                out.append(f"      {gene:<20} n={int(row['size']):<4} "
                           f"mean={row['mean']:+.4f}")

    out += ["", "  DOES TRANS CROSSTALK PREDICT COOPERATIVITY?"]
    tests = [("per gene, ATAC", trans_by_gene, coop_atac_gene, "gene"),
             ("per locus, ATAC", trans_by_locus, coop_atac_locus, "perturbed"),
             ("per gene, RNA", trans_by_gene, enh_coop, "gene")]
    any_run = False
    for label, trans, coop_table, key in tests:
        if trans is None or coop_table is None or trans.empty or coop_table.empty:
            continue
        if "cooperativity" not in coop_table.columns:
            continue
        merged = trans.merge(coop_table, left_on=key, right_on="name",
                             how="inner").dropna(subset=["mean_trans",
                                                         "cooperativity"])
        if len(merged) < 3:
            out.append(f"    {label:<20} only {len(merged)} matched group(s)")
            continue
        any_run = True
        out.append(_corr(merged["mean_trans"], merged["cooperativity"],
                         f"    {label}"))
    if not any_run:
        out.append("    [not supplied: trans_by_gene / coop tables]")

    out += ["", "  PER-PAIR CROSSTALK VS INTERACTION:"]
    if pair_comp is None or pair_comp.empty or \
            "trans_strongest" not in pair_comp.columns:
        out.append("    [not supplied: pair_comp after join_pair_trans]")
    else:
        n_attached = int(pair_comp["trans_strongest"].notna().sum())
        out.append(f"    pairs with trans statistics: {n_attached}/"
                   f"{len(pair_comp)}")
        for trans_col, value_col, label in [
                ("trans_strongest", "interaction_beta", "strongest vs measured"),
                ("trans_strongest", "coef", "strongest vs in silico"),
                ("trans_asymmetry", "interaction_beta", "asymmetry vs measured"),
                ("trans_asymmetry", "coef", "asymmetry vs in silico"),
                ("trans_mean", "interaction_beta", "mean vs measured")]:
            if trans_col in pair_comp.columns and value_col in pair_comp.columns:
                out.append(_corr(pair_comp[trans_col], pair_comp[value_col],
                                 f"    {label}"))
        out.append("    (crosstalk is negative and sub-additivity positive, so")
        out.append("     the hypothesis predicts a NEGATIVE correlation for")
        out.append("     strongest, and a POSITIVE one for asymmetry)")

    out += ["", "4.7  MODEL COMPARISON"]
    if comparison is None or comparison.empty:
        out += _missing("comparison")
    else:
        out.append(f"      {'setting':<22}{'n':>5}{'failed':>8}"
                   f"{'mean_se':>12}{'n_terms':>10}")
        for setting, group in comparison.groupby("setting"):
            out.append(f"      {setting:<22}{len(group):>5}"
                       f"{int((~group['converged']).sum()):>8}"
                       f"{group['mean_se'].mean():>12.4f}"
                       f"{group['n_terms'].mean():>10.1f}")
    return out


def _section_flags(motifs: pd.DataFrame | None, controls: pd.DataFrame | None,
                   pairs: pd.DataFrame | None, coop_atac: pd.DataFrame | None,
                   split: pd.DataFrame | None, floor_df: pd.DataFrame | None,
                   enh_inters: pd.DataFrame | None, single_comp: pd.DataFrame | None,
                   pair_comp: pd.DataFrame | None, comparison: pd.DataFrame | None,
                   alpha: float) -> list[str]:
    """Section 5: automatic flags for patterns to check before interpretation."""
    out = ["", _RULE, "5.  FLAGS", _RULE]
    flags = []

    if controls is None or (isinstance(controls, pd.DataFrame) and controls.empty):
        flags.append("CRITICAL: no specificity control. Motif effects cannot be "
                     "distinguished from generic sequence perturbation.")
    elif motifs is not None and not motifs.empty:
        threshold = np.percentile(controls["delta"].dropna(), 5)
        fraction = (motifs["delta"] < threshold).mean()
        if fraction < 0.80:
            flags.append(f"Only {fraction:.0%} of motifs fall below the control "
                         f"5th percentile. Specificity is weaker than ideal.")

    if split is None or (isinstance(split, pd.DataFrame) and split.empty):
        flags.append("No split-half saturation check. The naive diagnostic is "
                     "mathematically coupled and cannot be interpreted alone.")
    else:
        r = stats.pearsonr(split["additive_split"].dropna(),
                           split["interaction_split"].dropna())[0] \
            if split["additive_split"].notna().sum() > 2 else np.nan
        if np.isfinite(r) and abs(r) > 0.4:
            flags.append(f"Decoupled saturation r={r:+.2f}. Effect magnitudes "
                         f"depend on predicted loss; treat sizes as uncalibrated "
                         f"and rely on sign and rank.")

    if floor_df is not None and not floor_df.empty:
        fraction = floor_df["null_below_floor"].mean()
        if fraction > 0.10:
            flags.append(f"{fraction:.0%} of pairs have a null prediction below "
                         f"the readout floor. Their interaction terms are not "
                         f"estimable.")

    for label, table, col in [("TF pairs", pairs, "interaction"),
                              ("enhancer pairs", enh_inters, "coef")]:
        if table is None or table.empty or "qval" not in table.columns:
            continue
        n_sig = int((table["qval"] < alpha).sum())
        if n_sig < 10:
            flags.append(f"Only {n_sig} {label} survive FDR correction. "
                         f"Limit claims accordingly.")
        values = table[col].dropna()
        if len(values) and min((values < 0).mean(), (values > 0).mean()) < 0.10:
            direction = "sub-additive" if (values > 0).mean() > 0.9 else "super-additive"
            flags.append(f"{label} are almost entirely {direction} "
                         f"({max((values<0).mean(),(values>0).mean()):.0%}). "
                         f"A one-sided result often indicates a systematic cause.")

    if coop_atac is not None and not coop_atac.empty and \
            "n_seqlets" in coop_atac.columns:
        small = coop_atac.groupby("n_seqlets").size()
        thin = small[small < 5]
        if len(thin):
            readable = ", ".join(f"{int(k)} motifs: {int(v)} enhancer(s)"
                                 for k, v in thin.items())
            flags.append(f"Underpowered motif-count groups ({readable}). "
                         f"Group tests on these are not informative.")

    for label, comp in [("singles", single_comp), ("pairs", pair_comp)]:
        if comp is None or comp.empty:
            continue
        if len(comp) < 20:
            flags.append(f"CRISPRi validation for {label} rests on n={len(comp)}. "
                         f"Report the count alongside any correlation.")

    if comparison is not None and not comparison.empty:
        failed = comparison.groupby("setting")["converged"].apply(lambda s: (~s).sum())
        for setting, count in failed[failed > 0].items():
            flags.append(f"Model '{setting}' failed to converge for {count} gene(s).")

    if not flags:
        out.append("  No automatic flags raised.")
    else:
        for i, flag in enumerate(flags, start=1):
            out.append(f"  [{i}] {flag}")
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def build_summary(
    *,
    motifs: pd.DataFrame | None = None,
    controls: pd.DataFrame | None = None,
    pairs: pd.DataFrame | None = None,
    coop_atac: pd.DataFrame | None = None,
    coop_rna: pd.DataFrame | None = None,
    split: pd.DataFrame | None = None,
    floor_df: pd.DataFrame | None = None,
    dose: pd.DataFrame | None = None,
    enh_mains: pd.DataFrame | None = None,
    enh_inters: pd.DataFrame | None = None,
    atac_inters: pd.DataFrame | None = None,
    enh_coop: pd.DataFrame | None = None,
    single_comp: pd.DataFrame | None = None,
    pair_comp: pd.DataFrame | None = None,
    interdependency: pd.DataFrame | None = None,
    comparison: pd.DataFrame | None = None,
    cross: pd.DataFrame | None = None,
    validation_summaries: dict | None = None,
    trans_by_gene: pd.DataFrame | None = None,
    trans_by_locus: pd.DataFrame | None = None,
    coop_atac_gene: pd.DataFrame | None = None,
    coop_atac_locus: pd.DataFrame | None = None,
    alpha: float = 0.05,
    top_n: int = 10,
    title: str = "K562 enhancer and TF cooperativity",
) -> str:
    """Assemble the text summary from whichever tables are supplied.

    Args:
        motifs: TF motif effects.
        controls: Motif effects of the control windows.
        pairs: Pairwise TF motif interactions.
        coop_atac: TF cooperativity per enhancer on ATAC.
        coop_rna: TF cooperativity per enhancer on RNA.
        split: Split-half saturation table.
        floor_df: Floor check table.
        dose: Dose-response table.
        enh_mains: Enhancer-level GLM main effects.
        enh_inters: Enhancer-level GLM interactions.
        atac_inters: Enhancer-level interactions on ATAC.
        enh_coop: Additivity per gene.
        single_comp: Single-enhancer comparison with CRISPRi.
        pair_comp: Enhancer-pair comparison with CRISPRi.
        interdependency: Cis and trans accessibility effects.
        comparison: Model family and order comparison.
        cross: Merged cross-level table.
        validation_summaries: Mapping of level to summarise_validation output.
        trans_by_gene: Trans effects per target gene.
        trans_by_locus: Trans effects per perturbed enhancer.
        coop_atac_gene: Gene-level cooperativity on ATAC.
        coop_atac_locus: Locus-level cooperativity on ATAC.
        alpha: Significance threshold.
        top_n: Number of named extremes per direction.
        title: Document title.

    Returns:
        The summary text.
    """
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [_RULE, f"{title}", f"generated {stamp}",
             f"alpha={alpha}   top_n={top_n}", _RULE]

    supplied = {k: v for k, v in dict(
        motifs=motifs, controls=controls, pairs=pairs, coop_atac=coop_atac,
        coop_rna=coop_rna, split=split, floor_df=floor_df, dose=dose,
        enh_mains=enh_mains, enh_inters=enh_inters, atac_inters=atac_inters,
        enh_coop=enh_coop, single_comp=single_comp, pair_comp=pair_comp,
        interdependency=interdependency, comparison=comparison, cross=cross,
        trans_by_gene=trans_by_gene, trans_by_locus=trans_by_locus,
        coop_atac_gene=coop_atac_gene, coop_atac_locus=coop_atac_locus,
    ).items()}

    lines += ["", _THIN, "1.  INPUTS SUPPLIED", _THIN]
    for name, table in supplied.items():
        if table is None:
            lines.append(f"  {name:<20} MISSING")
        elif isinstance(table, pd.DataFrame):
            lines.append(f"  {name:<20} {len(table):>6} rows x "
                         f"{len(table.columns):>3} cols")

    lines += ["", _THIN, "2.  READING THIS DOCUMENT", _THIN,
              "  Sign convention: effects are log fold changes against the",
              "  unperturbed sequence, so scrambling gives a negative value.",
              "  Interaction I = observed minus additive prediction, therefore",
              "  I < 0 is SUPER-ADDITIVE and I > 0 is SUB-ADDITIVE.",
              "  Replicates are dinucleotide shuffles, which are technical,",
              "  not biological. p-values describe robustness to the shuffle",
              "  procedure only.",
              "  TF level uses predicted ATAC unless stated; enhancer level",
              "  uses predicted RNA at the target gene."]

    lines += ["", _RULE, "3.  TF LEVEL (within enhancer)", _RULE]
    lines += _section_tf_motifs(motifs, controls, alpha, top_n)
    lines += _section_tf_pairs(pairs, alpha, top_n)
    lines += _section_tf_coop(coop_atac, coop_rna, alpha, top_n)
    lines += _section_tf_artefacts(pairs, split, floor_df, dose)

    lines += _section_enhancer(enh_mains, enh_inters, atac_inters, enh_coop,
                               single_comp, pair_comp, interdependency,
                               comparison, validation_summaries, trans_by_gene,
                               trans_by_locus, coop_atac_gene, coop_atac_locus,
                               alpha, top_n)

    lines += ["", _RULE, "4.8  CROSS LEVEL", _RULE]
    if cross is None or (isinstance(cross, pd.DataFrame) and cross.empty):
        lines += _missing("cross")
    elif coop_atac is not None and enh_mains is not None:
        merged = cross
        numeric = merged.select_dtypes("number").columns.tolist()
        lines.append(f"  rows={len(merged)}  numeric columns={numeric[:6]}")
        if len(numeric) >= 2:
            lines.append(_corr(merged[numeric[0]], merged[numeric[1]],
                               f"{numeric[0]} vs {numeric[1]}"))

    lines += _section_flags(motifs, controls, pairs, coop_atac, split, floor_df,
                            enh_inters, single_comp, pair_comp, comparison,
                            alpha)

    lines += ["", _RULE, "END", _RULE]
    return "\n".join(lines)


def write_summary(path: str, **kwargs: Any) -> str:
    """Build the summary and write it to a file.

    Args:
        path: Output file.
        **kwargs: Passed to build_summary.

    Returns:
        The summary text.
    """
    text = build_summary(**kwargs)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    print(f"Wrote {path} ({len(text.splitlines())} lines, "
          f"{len(text) / 1024:.1f} KB)")
    return text