"""Plain-text summary of the cross-enhancer TF motif analysis.

Covers design and identifiability, motif main effects, within- versus
across-enhancer pairs, factor grammar, distance, accessibility interdependency,
the link to enhancer-level interactions, the within-enhancer run and exemplar
loci. Entry points: build_cross_summary, write_cross_summary.
"""

from __future__ import annotations

import datetime as _dt
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

# Formatting primitives are shared with the enhancer-level document so that the
# two summaries can be read side by side without relearning the layout.
from element_epistasis.report import (
    _RULE, _THIN, _corr, _describe, _direction_split, _fdr_line, _fmt_p,
    _missing, _top,
)

# A single colon separates the two units of an interaction term; a double colon
# belongs to a dimer motif name such as MAFG::NRF1.
_SINGLE_COLON = re.compile(r"(?<!:):(?!:)")

_CIS_CLASSES = ("cis", "mixed", "trans")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _has(df: Any, *columns: str) -> bool:
    """True when df is a non-empty DataFrame with every named column."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return False
    return set(columns).issubset(df.columns)


def _factor(label: Any) -> str:
    """Reduce a seqlet label to its factor name ("enh1|GATA1#2" -> "GATA1")."""
    text = str(label)
    if "|" in text:
        text = text.split("|")[-1]
    return text.split("#")[0]


def _factor_pair(row: pd.Series, a_col: str = "motif_a", b_col: str = "motif_b") -> str:
    """Order-independent factor pair label, e.g. "GATA1 x TAL1"."""
    first, second = sorted([_factor(row[a_col]), _factor(row[b_col])])
    return f"{first} x {second}"


def _pair_key(frame: pd.DataFrame, a_col: str, b_col: str) -> pd.Series:
    """Order-independent key for a pair of identifier columns."""
    return frame.apply(
        lambda r: "||".join(sorted([str(r[a_col]), str(r[b_col])])), axis=1)


def _bh(pvals: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values for one family, missing values kept."""
    out = pd.Series(np.nan, index=pvals.index, dtype=float)
    valid = pvals.notna()
    if not valid.any():
        return out

    values = pvals[valid].to_numpy()
    order = np.argsort(values)
    ranked = values[order]
    n = len(ranked)

    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    restored = np.empty(n)
    restored[order] = adjusted
    out.loc[valid] = restored
    return out


def _sign_test(values: pd.Series, label: str) -> str:
    """Binomial test of the negative/positive split against an even split, as one line."""
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    values = values[values != 0]
    if len(values) < 5:
        return f"  {label:<34} n={len(values)}, too few for a sign test"
    n_super = int((values < 0).sum())
    result = stats.binomtest(n_super, len(values), 0.5)
    return (f"  {label:<34} {n_super}/{len(values)} super-additive "
            f"({n_super / len(values):.1%})  {_fmt_p(result.pvalue)}")


def _group_line(values: pd.Series, label: str) -> str:
    """One line with n, median, IQR and direction split."""
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if values.empty:
        return f"      {label:<16}{'0':>7}{'-':>12}{'-':>22}{'-':>12}"
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    return (f"      {label:<16}{len(values):>7}{med:>+12.4f}"
            f"{f'[{q1:+.4f},{q3:+.4f}]':>22}"
            f"{(values < 0).mean():>12.1%}")


def _mannwhitney(first: pd.Series, second: pd.Series, label: str) -> str:
    """Two-sided Mann-Whitney U test between two value sets, as one line."""
    first = pd.to_numeric(pd.Series(first), errors="coerce").dropna()
    second = pd.to_numeric(pd.Series(second), errors="coerce").dropna()
    if len(first) < 3 or len(second) < 3:
        return (f"  {label:<34} not estimable "
                f"(n={len(first)}, {len(second)})")
    result = stats.mannwhitneyu(first, second, alternative="two-sided")
    return (f"  {label:<34} n={len(first)} vs {len(second)}  "
            f"{_fmt_p(result.pvalue)}")


def _split_mains(beta: pd.DataFrame, term_col: str = "term") -> pd.DataFrame:
    """Main-effect rows of a coefficient table with enhancer and motif split out."""
    if not _has(beta, term_col):
        return pd.DataFrame()
    out = beta[~beta[term_col].astype(str).str.contains(_SINGLE_COLON)].copy()
    out = out[~out[term_col].isin(["intercept", "alpha"])]
    if out.empty:
        return out
    has_sep = out[term_col].astype(str).str.contains("|", regex=False)
    out["enh"] = np.where(has_sep,
                          out[term_col].astype(str).str.split("|").str[0],
                          out[term_col])
    out["motif"] = np.where(has_sep,
                            out[term_col].astype(str).str.split("|").str[-1], "")
    out["factor"] = out["motif"].map(_factor)
    if "gene" in out.columns:
        out["full_unit"] = out["gene"] + "_" + out[term_col].astype(str)
    return out.reset_index(drop=True)


def _attach_pair_mains(inters: pd.DataFrame, beta: pd.DataFrame,
                       value_col: str = "coef") -> pd.DataFrame:
    """Add the two main effects (main_a, main_b) to each interaction row."""
    out = inters.copy()
    mains = _split_mains(beta)
    if mains.empty or "full_unit" not in mains.columns:
        out["main_a"] = np.nan
        out["main_b"] = np.nan
        return out

    lookup = mains.drop_duplicates("full_unit").set_index("full_unit")[value_col]
    out["main_a"] = out["full_a"].map(lookup)
    out["main_b"] = out["full_b"].map(lookup)
    return out


# --------------------------------------------------------------------------- #
# Section 3: design, coverage, identifiability
# --------------------------------------------------------------------------- #

def _section_design(inters_rna: pd.DataFrame | None, beta_rna: pd.DataFrame | None,
                    glm_results_rna: dict | None, glm_results_atac: dict | None,
                    seqlets: pd.DataFrame | None) -> list[str]:
    """Section 3: design size, coverage and GLM fit diagnostics."""
    out = ["", _THIN, "3.  DESIGN, COVERAGE AND IDENTIFIABILITY", _THIN]

    if not _has(inters_rna, "gene", "pair_scope"):
        out += _missing("inters_rna")
    else:
        scope_counts = inters_rna["pair_scope"].value_counts()
        out.append(f"  genes={inters_rna['gene'].nunique()}  "
                   f"motif pairs={len(inters_rna)}  "
                   f"within={int(scope_counts.get('within', 0))}  "
                   f"across={int(scope_counts.get('across', 0))}")
        if _has(inters_rna, "enh_pair"):
            across = inters_rna[inters_rna["pair_scope"] == "across"]
            out.append(f"  enhancer pairs with across-scope motif pairs="
                       f"{across['enh_pair'].nunique()}")
        per_gene = inters_rna.groupby("gene")["pair_scope"].value_counts().unstack(
            fill_value=0)
        for scope in ("within", "across"):
            if scope in per_gene.columns:
                out.append(_describe(per_gene[scope],
                                     f"{scope} pairs per gene"))
        both = per_gene.get("within", pd.Series(dtype=int)).gt(0) & \
            per_gene.get("across", pd.Series(dtype=int)).gt(0)
        out.append(f"  genes contributing both scopes: {int(both.sum())} of "
                   f"{len(per_gene)}")

    mains = _split_mains(beta_rna) if beta_rna is not None else pd.DataFrame()
    if not mains.empty:
        per_gene_units = mains.groupby("gene")["term"].nunique()
        out.append(_describe(per_gene_units, "scrambled units per gene"))
        out.append(f"  distinct factors={mains['factor'].nunique()}  "
                   f"enhancers carrying units="
                   f"{mains['full_unit'].str.rsplit('|', n=1).str[0].nunique()}")

    if seqlets is not None and isinstance(seqlets, pd.DataFrame) \
            and not seqlets.empty:
        out.append(f"  seqlet table supplied: {len(seqlets)} rows")

    out += ["", "  fit diagnostics:"]
    for label, results in [("RNA, per gene", glm_results_rna),
                           ("ATAC, per locus", glm_results_atac)]:
        if not results:
            out.append(f"    {label:<18} [not supplied]")
            continue
        total = len(results)
        converged = sum(1 for r in results.values() if r.get("converged"))
        deficient = [k for k, r in results.items()
                     if (r.get("identifiability") or {}).get("full_rank") is False]
        n_obs = [r.get("n_obs") for r in results.values() if r.get("n_obs")]
        out.append(f"    {label:<18} fits={total}  converged={converged}  "
                   f"rank deficient={len(deficient)}  "
                   f"median n_obs={int(np.median(n_obs)) if n_obs else 0}")
        if deficient:
            out.append(f"      deficient: {', '.join(sorted(deficient)[:6])}"
                       f"{' ...' if len(deficient) > 6 else ''}")
            out.append("      Coefficients for unestimable terms are returned "
                       "by statsmodels regardless and are not interpretable.")
    return out


# --------------------------------------------------------------------------- #
# Section 4: motif main effects
# --------------------------------------------------------------------------- #

def _section_main_effects(beta_rna: pd.DataFrame | None, inters_rna: pd.DataFrame | None,
                          alpha: float, top_n: int) -> list[str]:
    """Section 4: motif main effects on expression."""
    out = ["", _THIN, "4.  MOTIF MAIN EFFECTS ON EXPRESSION (beta_i)", _THIN]
    mains = _split_mains(beta_rna) if beta_rna is not None else pd.DataFrame()
    if mains.empty:
        return out + _missing("beta_rna main effects")

    out.append(f"  terms={len(mains)}  genes={mains['gene'].nunique()}  "
               f"factors={mains['factor'].nunique()}")
    out.append(_describe(mains["coef"], "beta_i"))
    out.append(_fdr_line(mains, "significant motifs", alpha=alpha))
    out.append(_sign_test(mains["coef"], "direction of beta_i"))

    # Scrambling a functional motif should reduce expression, so the fraction
    # of negative main effects is a coarse check that the readout responds.
    negative = (mains["coef"] < 0).mean()
    out.append(f"  fraction with beta_i < 0: {negative:.1%}")

    if _has(inters_rna, "coef"):
        median_main = mains["coef"].abs().median()
        median_inter = inters_rna["coef"].abs().median()
        if median_inter > 0:
            out.append(f"  median |beta_i| = {median_main:.4f}, "
                       f"median |beta_ij| = {median_inter:.4f}, "
                       f"ratio = {median_main / median_inter:.1f}x")

    per_factor = (mains.groupby("factor")["coef"]
                  .agg(n="size", mean="mean", median="median")
                  .sort_values("mean"))
    shown = min(len(per_factor), 25)
    out += ["", f"  per factor (sorted by mean effect, showing {shown} of "
                f"{len(per_factor)}):",
            f"      {'factor':<20}{'n':>5}{'mean':>10}{'median':>10}"]
    for factor, row in per_factor.head(shown).iterrows():
        out.append(f"      {str(factor):<20}{int(row['n']):>5}"
                   f"{row['mean']:>+10.4f}{row['median']:>+10.4f}")

    out += ["", "  strongest individual motif effects:"]
    out += _top(mains, "coef", ["gene", "term"], top_n,
                both_directions=False, extra_cols=["qval"])
    return out


# --------------------------------------------------------------------------- #
# Section 5: interactions by scope
# --------------------------------------------------------------------------- #

def _section_scope(inters_rna: pd.DataFrame | None, beta_rna: pd.DataFrame | None, scope: dict | None,
                   alpha: float, top_n: int) -> list[str]:
    """Section 5: within- versus across-enhancer pair interactions."""
    out = ["", _RULE,
           "5.  MOTIF PAIR INTERACTIONS BY SCOPE (beta_ij, expression)", _RULE]
    if not _has(inters_rna, "coef", "pair_scope"):
        return out + _missing("inters_rna")

    work = inters_rna.dropna(subset=["coef"]).copy()
    within = work[work["pair_scope"] == "within"]["coef"]
    across = work[work["pair_scope"] == "across"]["coef"]

    out += ["", "5.1  DISTRIBUTIONS",
            f"      {'scope':<16}{'n':>7}{'median':>12}{'IQR':>22}"
            f"{'super-add':>12}"]
    out.append(_group_line(within, "within"))
    out.append(_group_line(across, "across"))
    out.append(_group_line(work["coef"], "all"))

    out.append("")
    for label, values in [("within", within), ("across", across)]:
        out.append(_sign_test(values, f"direction, {label}"))
        out.append(_describe(values.abs(), f"|beta_ij|, {label}"))

    out.append("")
    out.append(_mannwhitney(within, across, "within vs across, signed"))
    out.append(_mannwhitney(within.abs(), across.abs(),
                            "within vs across, magnitude"))
    if isinstance(scope, dict) and scope:
        out.append(f"  supplied scope summary: "
                   f"signed {_fmt_p(scope.get('mannwhitney_p'))}, "
                   f"magnitude {_fmt_p(scope.get('mannwhitney_p_abs'))}")

    if "qval" in work.columns:
        out += ["", "5.2  SIGNIFICANCE"]
        for label in ("within", "across"):
            block = work[work["pair_scope"] == label]
            out.append(_fdr_line(block, f"significant, {label}", alpha=alpha))
        # BH was applied to whichever family the caller passed. Pooling main
        # effects with interactions lowers the effective threshold for the
        # interactions, so the within-family recomputation is the honest count.
        if "pval" in work.columns:
            refit = _bh(work["pval"])
            out.append(f"  {'BH within interactions only':<34} "
                       f"{int((refit < alpha).sum())} of "
                       f"{int(refit.notna().sum())} at FDR<{alpha:g}")
            for label in ("within", "across"):
                mask = work["pair_scope"] == label
                out.append(f"      {label:<10} "
                           f"{int((refit[mask] < alpha).sum())} of "
                           f"{int(mask.sum())}")

    out += ["", "5.3  PER GENE, PAIRED"]
    out += _scope_paired(work)

    out += ["", "5.4  SATURATION COUPLING"]
    out += _scope_saturation(work, beta_rna)

    out += ["", "5.5  HOMOTYPIC VERSUS HETEROTYPIC PAIRS"]
    out += _scope_homotypic(work)

    out += ["", "5.6  EXTREME PAIRS"]
    for label in ("within", "across"):
        block = work[work["pair_scope"] == label]
        out.append(f"    scope: {label}")
        out += _top(block, "coef", ["gene", "term"], top_n,
                    extra_cols=["qval", "distance_bp"])
    return out


def _scope_paired(work: pd.DataFrame) -> list[str]:
    """Per-gene paired comparison of the two scopes."""
    out = []
    if "gene" not in work.columns:
        return ["  gene column absent"]

    medians = (work.pivot_table(index="gene", columns="pair_scope",
                                values="coef", aggfunc="median"))
    if not {"within", "across"}.issubset(medians.columns):
        return ["  only one scope present; paired comparison not possible"]

    paired = medians.dropna(subset=["within", "across"])
    out.append(f"  genes with both scopes: {len(paired)}")
    if len(paired) < 5:
        out.append("  too few genes for a paired test")
        return out

    difference = paired["within"] - paired["across"]
    signed = stats.wilcoxon(difference)
    out.append(f"  median per-gene within = {paired['within'].median():+.4f}, "
               f"across = {paired['across'].median():+.4f}")
    out.append(f"  median difference (within minus across) = "
               f"{difference.median():+.4f}  "
               f"Wilcoxon {_fmt_p(signed.pvalue)}")
    out.append(f"  genes where within is more negative: "
               f"{int((difference < 0).sum())}/{len(difference)}")
    out.append("  This is the scope contrast controlled for gene identity; "
               "the pooled test above is not.")
    return out


def _scope_saturation(work: pd.DataFrame, beta_rna: pd.DataFrame | None) -> list[str]:
    """Correlation of interaction size with the summed main effects, per scope."""
    out = []
    if beta_rna is None:
        return ["  [not supplied: beta_rna] coupling is UNTESTED"]

    attached = _attach_pair_mains(work, beta_rna)
    if attached["main_a"].isna().all():
        return ["  main effects could not be matched to pairs"]

    additive = attached["main_a"] + attached["main_b"]
    out.append(_corr(additive, attached["coef"], "beta_ij vs beta_i + beta_j"))
    out.append(_corr(additive.abs(), attached["coef"].abs(),
                     "|beta_ij| vs |beta_i + beta_j|"))
    for label in ("within", "across"):
        mask = attached["pair_scope"] == label
        out.append(_corr(additive[mask], attached.loc[mask, "coef"],
                         f"  restricted to {label}"))
    out.append("  A NEGATIVE slope is the signature of a floor or of predictor")
    out.append("  compression; a positive one is not explained by either.")
    out.append("  This diagnostic is mathematically coupled, since beta_ij and")
    out.append("  the main effects are estimated from overlapping observations.")
    return out


def _scope_homotypic(work: pd.DataFrame) -> list[str]:
    """Same-factor versus different-factor pairs."""
    out = []
    if not {"motif_a", "motif_b"}.issubset(work.columns):
        return ["  motif columns absent"]

    labelled = work.copy()
    labelled["fa"] = labelled["motif_a"].map(_factor)
    labelled["fb"] = labelled["motif_b"].map(_factor)
    labelled["homotypic"] = labelled["fa"] == labelled["fb"]

    out.append(f"      {'class':<16}{'n':>7}{'median':>12}{'IQR':>22}"
               f"{'super-add':>12}")
    for flag, label in [(True, "homotypic"), (False, "heterotypic")]:
        out.append(_group_line(labelled.loc[labelled["homotypic"] == flag,
                                            "coef"], label))
    out.append("")
    out.append(_mannwhitney(labelled.loc[labelled["homotypic"], "coef"],
                            labelled.loc[~labelled["homotypic"], "coef"],
                            "homotypic vs heterotypic"))
    return out


# --------------------------------------------------------------------------- #
# Section 6: grammar
# --------------------------------------------------------------------------- #

def _section_grammar(inters_rna: pd.DataFrame | None, alpha: float, top_n: int) -> list[str]:
    """Section 6: recurrent factor pairs."""
    out = ["", _THIN, "6.  RECURRENT FACTOR GRAMMAR", _THIN]
    if not _has(inters_rna, "coef", "motif_a", "motif_b"):
        return out + _missing("inters_rna")

    work = inters_rna.dropna(subset=["coef"]).copy()
    work["factor_pair"] = work.apply(_factor_pair, axis=1)
    work["_sig"] = (work["qval"] < alpha) if "qval" in work.columns else False

    for scope in ("within", "across"):
        block = work[work["pair_scope"] == scope] \
            if "pair_scope" in work.columns else work
        agg = (block.groupby("factor_pair")
                    .agg(n=("coef", "size"), mean=("coef", "mean"),
                         median=("coef", "median"), sig=("_sig", "sum"))
                    .query("n >= 2").sort_values("mean"))
        out += ["", f"  {scope}-enhancer factor pairs seen at least twice "
                    f"(n={len(agg)}):"]
        if agg.empty:
            out.append("      (none recur)")
            continue
        out.append(f"      {'factor pair':<30}{'n':>5}{'mean':>11}"
                   f"{'median':>11}{'sig':>6}")
        for pair, row in agg.head(top_n * 2).iterrows():
            out.append(f"      {str(pair):<30}{int(row['n']):>5}"
                       f"{row['mean']:>+11.4f}{row['median']:>+11.4f}"
                       f"{int(row['sig']):>6}")
        if len(agg) > top_n * 2:
            out.append(f"      ... {len(agg) - top_n * 2} further pairs")

    # A factor-level marginal answers a different question from the pair table:
    # whether one factor tends towards a direction whatever it is paired with.
    marginal = pd.concat([
        work.assign(factor=work["motif_a"].map(_factor))[["factor", "coef"]],
        work.assign(factor=work["motif_b"].map(_factor))[["factor", "coef"]],
    ])
    per_factor = (marginal.groupby("factor")["coef"]
                  .agg(n="size", mean="mean", median="median")
                  .query("n >= 3").sort_values("mean"))
    out += ["", f"  per factor across all its pairs (n>=3, {len(per_factor)} "
                f"factors):"]
    if per_factor.empty:
        out.append("      (none)")
    else:
        out.append(f"      {'factor':<20}{'n':>5}{'mean':>11}{'median':>11}")
        for factor, row in per_factor.head(top_n * 2).iterrows():
            out.append(f"      {str(factor):<20}{int(row['n']):>5}"
                       f"{row['mean']:>+11.4f}{row['median']:>+11.4f}")
    return out


# --------------------------------------------------------------------------- #
# Section 7: distance
# --------------------------------------------------------------------------- #

def _section_distance(inters_rna: pd.DataFrame | None) -> list[str]:
    """Section 7: interaction versus motif distance."""
    out = ["", _THIN, "7.  DISTANCE DEPENDENCE", _THIN]
    if not _has(inters_rna, "coef", "distance_bp"):
        return out + _missing("inters_rna with distance_bp")

    work = inters_rna.dropna(subset=["coef", "distance_bp"])
    work = work[work["distance_bp"] > 0]
    if work.empty:
        return out + ["  no positive distances"]

    out.append(f"  pairs with a distance: {len(work)} of {len(inters_rna)}")
    out.append(_describe(work["distance_bp"], "distance (bp)"))
    out.append(_corr(work["distance_bp"], work["coef"], "distance vs beta_ij"))
    out.append(_corr(np.log10(work["distance_bp"]), work["coef"],
                     "log10 distance vs beta_ij"))

    out += ["", "  within each scope, since scope determines the distance range",
            "  and the pooled correlation is confounded by it:"]
    for scope in ("within", "across"):
        block = work[work["pair_scope"] == scope] \
            if "pair_scope" in work.columns else work.iloc[0:0]
        out.append(_corr(block["distance_bp"], block["coef"], f"  {scope}"))
        if len(block):
            out.append(_describe(block["distance_bp"], f"  {scope} distance"))

    edges = [0, 1e2, 1e3, 1e4, 1e5, np.inf]
    names = ["<100 bp", "100 bp to 1 kb", "1 to 10 kb", "10 to 100 kb",
             ">100 kb"]
    binned = pd.cut(work["distance_bp"], bins=edges, labels=names,
                    right=False)
    out += ["", f"      {'bin':<18}{'n':>7}{'median':>12}{'IQR':>22}"
                f"{'super-add':>12}"]
    for name in names:
        out.append(_group_line(work.loc[binned == name, "coef"], name))
    return out


# --------------------------------------------------------------------------- #
# Section 8: accessibility interdependency
# --------------------------------------------------------------------------- #

def _section_interdependency(interdep: pd.DataFrame | None, alpha: float,
                             top_n: int) -> list[str]:
    """Section 8: motif effects on accessibility by cis, mixed and trans class."""
    out = ["", _RULE, "8.  ACCESSIBILITY: MOTIF INTERDEPENDENCY", _RULE]
    if not _has(interdep, "delta", "level", "cis_class"):
        return out + _missing("interdep")

    mains = interdep[interdep["level"] == "main"]
    pairs = interdep[interdep["level"] == "interaction"]

    out += ["", "8.1  SINGLE MOTIFS (beta_i on accessibility)"]
    if mains.empty:
        out += _missing("main level rows")
    else:
        out.append(f"  cells={len(mains)}  genes={mains['gene'].nunique()}")
        out.append(f"      {'class':<16}{'n':>7}{'median':>12}{'IQR':>22}"
                   f"{'negative':>12}")
        for cls in ("cis", "trans"):
            out.append(_group_line(mains.loc[mains["cis_class"] == cls,
                                             "delta"], cls))
        cis_median = mains.loc[mains["cis_class"] == "cis", "delta"].median()
        trans_median = mains.loc[mains["cis_class"] == "trans", "delta"].median()
        if np.isfinite(cis_median) and np.isfinite(trans_median) \
                and trans_median != 0:
            out.append(f"  cis to trans median ratio: "
                       f"{cis_median / trans_median:.1f}x")
        if "qval" in mains.columns:
            for cls in ("cis", "trans"):
                block = mains[mains["cis_class"] == cls]
                out.append(_fdr_line(block, f"significant, {cls}", alpha=alpha))

    out += ["", "8.2  MOTIF PAIRS (beta_ij on accessibility)"]
    if pairs.empty:
        out += _missing("interaction level rows")
        return out

    out.append(f"  cells={len(pairs)}  genes={pairs['gene'].nunique()}")
    out.append(f"      {'class':<16}{'n':>7}{'median':>12}{'IQR':>22}"
               f"{'super-add':>12}")
    for cls in _CIS_CLASSES:
        out.append(_group_line(pairs.loc[pairs["cis_class"] == cls, "delta"],
                               cls))
    out.append("")
    out.append("  cis: both motifs inside the measured enhancer, local grammar")
    out.append("  mixed: one motif inside and one elsewhere, the direct test of")
    out.append("         whether a distant motif modulates local grammar")
    out.append("  trans: neither motif inside, both acting at a distance")

    groups = [pairs.loc[pairs["cis_class"] == cls, "delta"].dropna()
              for cls in _CIS_CLASSES]
    if all(len(g) >= 3 for g in groups):
        kruskal = stats.kruskal(*groups)
        out.append("")
        out.append(f"  Kruskal-Wallis across the three classes: "
                   f"{_fmt_p(kruskal.pvalue)}")
        for first, second in [("cis", "mixed"), ("cis", "trans"),
                              ("mixed", "trans")]:
            out.append(_mannwhitney(
                pairs.loc[pairs["cis_class"] == first, "delta"],
                pairs.loc[pairs["cis_class"] == second, "delta"],
                f"{first} vs {second}"))

    if "qval" in pairs.columns:
        out.append("")
        for cls in _CIS_CLASSES:
            block = pairs[pairs["cis_class"] == cls]
            out.append(_fdr_line(block, f"significant, {cls}", alpha=alpha))

    out += ["", "8.3  THE MIXED CLASS IN DETAIL"]
    mixed = pairs[pairs["cis_class"] == "mixed"]
    if mixed.empty:
        out += ["  no mixed-class cells"]
    else:
        out.append(_sign_test(mixed["delta"], "direction, mixed"))
        out.append(_describe(mixed["delta"].abs(), "|beta_ij|, mixed"))
        cis_abs = pairs.loc[pairs["cis_class"] == "cis", "delta"].abs()
        if len(cis_abs) >= 3:
            out.append(_mannwhitney(mixed["delta"].abs(), cis_abs,
                                    "mixed vs cis, magnitude"))
        out += ["", "  strongest mixed-class cells:"]
        out += _top(mixed, "delta", ["measured", "perturbed"], top_n,
                    extra_cols=["qval"])
    return out


# --------------------------------------------------------------------------- #
# Section 9: readout agreement
# --------------------------------------------------------------------------- #

def _perturbed_key(frame: pd.DataFrame) -> pd.Series:
    """Order-independent key naming the two perturbed units of a pair."""
    if {"full_a", "full_b"}.issubset(frame.columns):
        return _pair_key(frame, "full_a", "full_b")
    if "perturbed" in frame.columns:
        return frame["perturbed"].astype(str).apply(
            lambda value: "||".join(sorted(value.split(" x "))))
    return pd.Series(index=frame.index, dtype=object)


def _section_readouts(inters_rna: pd.DataFrame | None, inters_atac: pd.DataFrame | None, interdep: pd.DataFrame | None,
                      alpha: float) -> list[str]:
    """Section 9: expression versus accessibility interactions of the same pairs."""
    out = ["", _THIN, "9.  EXPRESSION VERSUS ACCESSIBILITY, SAME PAIRS", _THIN]
    if not _has(inters_rna, "coef"):
        return out + _missing("inters_rna")

    left = inters_rna.dropna(subset=["coef"]).copy()
    left["_key"] = _perturbed_key(left)
    left["_gene_key"] = left["gene"].astype(str) + "::" + left["_key"]

    if _has(interdep, "delta", "level"):
        right = interdep[interdep["level"] == "interaction"].copy()
        right["_key"] = _perturbed_key(right)
        right["_gene_key"] = right["gene"].astype(str) + "::" + right["_key"]
        merged = left.merge(
            right[["_gene_key", "measured", "cis_class", "delta"]],
            on="_gene_key", how="inner")
        if merged.empty:
            out.append("  no motif pair matched between the two readouts; "
                       "check that unit labels agree")
            return out

        out.append(f"  matched cells={len(merged)}  "
                   f"distinct pairs={merged['_gene_key'].nunique()}")
        out.append(_corr(merged["delta"], merged["coef"],
                         "ATAC beta_ij vs RNA beta_ij"))
        concordant = np.sign(merged["delta"]) == np.sign(merged["coef"])
        out.append(f"  sign concordant: {int(concordant.sum())}/{len(merged)} "
                   f"({concordant.mean():.1%})")
        out += ["", "  by relation of the pair to the measured enhancer:"]
        for cls in _CIS_CLASSES:
            block = merged[merged["cis_class"] == cls]
            out.append(_corr(block["delta"], block["coef"], f"  {cls}"))
        out += ["", "  Accessibility is measured at one enhancer and expression",
                "  at the gene, so agreement is expected only where the",
                "  measured enhancer is one that the pair acts through."]
        return out

    if _has(inters_atac, "coef", "gene", "term"):
        collapsed = (inters_atac.groupby(["gene", "term"])["coef"]
                     .median().rename("coef_atac").reset_index())
        merged = left.merge(collapsed, on=["gene", "term"], how="inner")
        out.append(f"  matched pairs={len(merged)} "
                   f"(accessibility taken as the median over measured loci)")
        if len(merged) >= 3:
            out.append(_corr(merged["coef_atac"], merged["coef"],
                             "ATAC beta_ij vs RNA beta_ij"))
            concordant = np.sign(merged["coef_atac"]) == np.sign(merged["coef"])
            out.append(f"  sign concordant: {int(concordant.sum())}/"
                       f"{len(merged)} ({concordant.mean():.1%})")
        return out

    return out + _missing("interdep or inters_atac")


# --------------------------------------------------------------------------- #
# Section 10: cross level
# --------------------------------------------------------------------------- #

def aggregate_pairs(inters_rna: pd.DataFrame, scope: str = "across",
                    value_col: str = "coef", alpha: float = 0.05
                    ) -> pd.DataFrame:
    """Aggregate motif-pair interactions to enhancer pairs, keeping the strongest motif pair.

    Args:
        inters_rna: Classified interaction terms.
        scope: Pair scope to keep, normally "across".
        value_col: Interaction coefficient column.
        alpha: Significance threshold for the count of significant pairs.

    Returns:
        One row per gene and enhancer pair.
    """
    required = {"gene", "enhA", "enhB", "enh_pair", "pair_scope", value_col}
    if not _has(inters_rna, *required):
        return pd.DataFrame()

    work = inters_rna[inters_rna["pair_scope"] == scope].dropna(
        subset=[value_col]).copy()
    if work.empty:
        return pd.DataFrame()

    work["_sig"] = (work["qval"] < alpha) if "qval" in work.columns else False
    work["_abs"] = work[value_col].abs()
    keys = ["gene", "enhA", "enhB", "enh_pair"]

    agg = (work.groupby(keys)
               .agg(motif_sum=(value_col, "sum"),
                    motif_mean=(value_col, "mean"),
                    motif_strongest=(value_col, "min"),
                    motif_max=(value_col, "max"),
                    n_motif_pairs=(value_col, "size"),
                    n_significant=("_sig", "sum"))
               .reset_index())

    # The lead pair is the largest in absolute value, which is the one a reader
    # would point at, rather than the most negative.
    lead = work.loc[work.groupby(keys)["_abs"].idxmax()].copy()
    lead_cols = {"motif_a": "lead_motif_a", "motif_b": "lead_motif_b",
                 "enh_a": "lead_enh_a", "enh_b": "lead_enh_b",
                 value_col: "lead_coef", "qval": "lead_qval",
                 "distance_bp": "lead_distance_bp", "term": "lead_term",
                 "full_a": "lead_full_a", "full_b": "lead_full_b"}
    keep = [c for c in lead_cols if c in lead.columns]
    lead = lead[keys + keep].rename(columns=lead_cols)
    return agg.merge(lead, on=keys, how="left")


def _enh_pair_key(enh_inters: pd.DataFrame) -> pd.Series:
    """Order-independent enhancer pair key for the enhancer-level table."""
    if {"full_a", "full_b"}.issubset(enh_inters.columns):
        return _pair_key(enh_inters, "full_a", "full_b")
    if {"enhA", "enhB"}.issubset(enh_inters.columns):
        return _pair_key(enh_inters, "enhA", "enhB")
    return pd.Series(index=enh_inters.index, dtype=object)


def _section_cross_level(inters_rna: pd.DataFrame | None, pair_agg: pd.DataFrame | None, enh_inters: pd.DataFrame | None,
                         enh_mains: pd.DataFrame | None, measured_pairs: pd.DataFrame | None,
                         alpha: float) -> list[str]:
    """Section 10: aggregated motif interactions versus enhancer-level and measured interactions."""
    out = ["", _RULE,
           "10.  DOES MOTIF GRAMMAR EXPLAIN THE ENHANCER INTERACTION", _RULE]
    if pair_agg is None or (isinstance(pair_agg, pd.DataFrame)
                            and pair_agg.empty):
        pair_agg = aggregate_pairs(inters_rna, alpha=alpha)
    if pair_agg is None or pair_agg.empty:
        return out + _missing("pair_agg")
    if not _has(enh_inters, "coef"):
        return out + _missing("enh_inters")

    left = pair_agg.copy()
    left["_key"] = _pair_key(left, "enhA", "enhB")
    right = enh_inters.copy()
    right["_key"] = _enh_pair_key(right)
    right = right.rename(columns={"coef": "enh_coef", "qval": "enh_qval",
                                  "distance_bp": "enh_distance_bp"})
    keep = [c for c in ["_key", "enh_coef", "enh_qval", "enh_distance_bp"]
            if c in right.columns]
    merged = left.merge(right[keep].drop_duplicates("_key"), on="_key",
                        how="inner").dropna(subset=["enh_coef"])

    out.append(f"  enhancer pairs on the motif side: {len(left)}")
    out.append(f"  enhancer pairs on the knockout side: "
               f"{right['_key'].nunique()}")
    out.append(f"  matched: {len(merged)}")
    if len(merged) < 3:
        out.append("  too few matched pairs for any statistic; check that "
                   "enhancer identifiers agree between the two tables")
        return out

    out.append(_describe(merged["n_motif_pairs"], "motif pairs per enhancer pair"))
    out += ["", f"      {'aggregation':<20}{'n':>5}{'r':>9}{'p':>12}"
                f"{'rho':>9}{'p':>12}{'sign agree':>12}"]
    for column in ["motif_mean", "motif_sum", "motif_strongest", "motif_max"]:
        if column not in merged.columns:
            continue
        block = merged.dropna(subset=[column, "enh_coef"])
        if len(block) < 3:
            out.append(f"      {column:<20}{len(block):>5}  not estimable")
            continue
        r, p_r = stats.pearsonr(block[column], block["enh_coef"])
        rho, p_rho = stats.spearmanr(block[column], block["enh_coef"])
        agree = (np.sign(block[column]) == np.sign(block["enh_coef"])).mean()
        out.append(f"      {column:<20}{len(block):>5}{r:>+9.3f}"
                   f"{p_r:>12.4f}{rho:>+9.3f}{p_rho:>12.4f}{agree:>12.1%}")
    out.append("  motif_mean is the primary aggregation; motif_sum grows with "
               "the number of")
    out.append("  motif pairs tested and is therefore confounded with seqlet "
               "coverage.")

    # A correlation carried by the number of motif pairs would be a coverage
    # artefact rather than evidence about grammar.
    if merged["n_motif_pairs"].nunique() < 2:
        out.append(f"  {'motif_mean | n_motif_pairs':<34} not estimable, every "
                   f"enhancer pair contributes the same number of motif pairs")
    else:
        try:
            from element_epistasis.tf_cooperativity import partial_correlation
            r, p_val, n = partial_correlation(merged["motif_mean"],
                                              merged["enh_coef"],
                                              merged["n_motif_pairs"])
            out.append(f"  {'motif_mean | n_motif_pairs':<34} n={n:<5} "
                       f"partial r={r:+.3f} {_fmt_p(p_val)}")
        except Exception as error:
            out.append(f"  partial correlation failed: {error}")

    if "enh_qval" in merged.columns:
        confident = merged[merged["enh_qval"] < alpha]
        out += ["", f"  restricted to enhancer pairs significant at the "
                    f"knockout level (n={len(confident)}):"]
        if len(confident) >= 3:
            out.append(_corr(confident["motif_mean"], confident["enh_coef"],
                             "  motif_mean vs beta_ij"))
        else:
            out.append("    too few")
    if "n_significant" in merged.columns:
        confident = merged[merged["n_significant"] > 0]
        out += ["", f"  restricted to enhancer pairs with at least one "
                    f"significant motif pair (n={len(confident)}):"]
        if len(confident) >= 3:
            out.append(_corr(confident["motif_mean"], confident["enh_coef"],
                             "  motif_mean vs beta_ij"))
        else:
            out.append("    too few")

    if _has(measured_pairs, "interaction_beta"):
        out += ["", "  against the measured CRISPRi interaction:"]
        screen = measured_pairs.copy()
        screen["_key"] = _enh_pair_key(screen)
        joined = merged.merge(
            screen[["_key", "interaction_beta"]].drop_duplicates("_key"),
            on="_key", how="inner").dropna(subset=["interaction_beta"])
        out.append(f"    matched pairs: {len(joined)}")
        if len(joined) >= 3:
            out.append(_corr(joined["motif_mean"], joined["interaction_beta"],
                             "    motif_mean vs measured"))
            agree = (np.sign(joined["motif_mean"])
                     == np.sign(joined["interaction_beta"])).mean()
            out.append(f"    sign concordant: {agree:.1%}")
    else:
        out += ["", "  [not supplied: measured_pairs] the screen comparison at "
                    "this level is untested"]
    return out


# --------------------------------------------------------------------------- #
# Section 11: contrast with the within-enhancer run
# --------------------------------------------------------------------------- #

def _section_within_run(inters_rna: pd.DataFrame | None, interdep: pd.DataFrame | None, tf_pairs: pd.DataFrame | None,
                        alpha: float) -> list[str]:
    """Section 11: comparison with the within-enhancer run."""
    out = ["", _THIN, "11.  CONTRAST WITH THE WITHIN-ENHANCER RUN", _THIN]
    if not _has(tf_pairs, "interaction", "enhancer", "motif_a", "motif_b"):
        return out + _missing("tf_pairs")

    earlier = tf_pairs.dropna(subset=["interaction"]).copy()
    earlier["_key"] = _pair_key(earlier, "motif_a", "motif_b")
    earlier["_enh_key"] = earlier["enhancer"].astype(str) + "::" + earlier["_key"]
    earlier["factor_pair"] = earlier.apply(_factor_pair, axis=1)

    out.append(f"  earlier run: pairs={len(earlier)}  "
               f"enhancers={earlier['enhancer'].nunique()}")
    out.append(_describe(earlier["interaction"], "I, earlier run (ATAC)"))

    out += ["", "  11.1  READOUT-MATCHED AND OBJECT-MATCHED COMPARISON",
            "  Accessibility measured at the enhancer that contains both",
            "  motifs, so the only difference from the earlier run is the",
            "  window anchoring."]
    if not _has(interdep, "delta", "level", "cis_class"):
        out += _missing("interdep")
    else:
        cis = interdep[(interdep["level"] == "interaction")
                       & (interdep["cis_class"] == "cis")].copy()
        if cis.empty:
            out.append("  no cis-class interaction cells")
        else:
            cis["_key"] = cis.apply(
                lambda r: "||".join(sorted([str(r.get("motif_a", "")),
                                            str(r.get("motif_b", ""))])), axis=1)
            cis["_enh_key"] = cis["gene"].astype(str) + "_" + \
                cis["enh_a"].astype(str) + "::" + cis["_key"]
            merged = cis.merge(
                earlier[["_enh_key", "interaction"]].drop_duplicates("_enh_key"),
                on="_enh_key", how="inner")
            out.append(f"  matched on exact seqlet labels: {len(merged)} of "
                       f"{len(cis)} cis cells")

            # Duplicate seqlets are numbered independently in the two runs, so
            # an exact label match can fail on objects that are in fact the
            # same. The factor-level match is the fallback, and is ambiguous
            # wherever one factor pair occurs twice in an enhancer.
            if len(merged) < len(cis):
                cis["_fkey"] = cis["gene"].astype(str) + "_" + \
                    cis["enh_a"].astype(str) + "::" + cis.apply(_factor_pair,
                                                                axis=1)
                earlier["_fkey"] = earlier["enhancer"].astype(str) + "::" + \
                    earlier["factor_pair"]
                loose = cis.drop_duplicates("_fkey").merge(
                    earlier.drop_duplicates("_fkey")[["_fkey", "interaction"]],
                    on="_fkey", how="inner")
                out.append(f"  matched on factor names only: {len(loose)} "
                           f"(ambiguous where a factor pair recurs)")
                if len(merged) < 3 <= len(loose):
                    merged = loose

            if len(merged) >= 3:
                out.append(_corr(merged["interaction"], merged["delta"],
                                 "earlier I vs this run beta_ij"))
                agree = (np.sign(merged["interaction"])
                         == np.sign(merged["delta"])).mean()
                out.append(f"  sign concordant: {agree:.1%}")
                out.append(_describe(merged["delta"] - merged["interaction"],
                                     "difference, this minus earlier"))
            elif len(merged) == 0:
                out.append("  labels did not match; the earlier run may name "
                           "enhancers or motifs differently")

    out += ["", "  11.2  DISTRIBUTION COMPARISON",
            "  Descriptive only: the earlier run used a different window and,",
            "  for the expression readout, a different response variable."]
    if _has(inters_rna, "coef", "pair_scope"):
        within = inters_rna[inters_rna["pair_scope"] == "within"]["coef"]
        out.append(_group_line(within, "this, within"))
        out.append(_group_line(earlier["interaction"], "earlier"))
        out.append(_mannwhitney(within, earlier["interaction"],
                                "this within vs earlier"))

    out += ["", "  11.3  FACTOR PAIRS SEEN AT BOTH LEVELS"]
    if _has(inters_rna, "motif_a", "motif_b", "coef"):
        current = inters_rna.dropna(subset=["coef"]).copy()
        current["factor_pair"] = current.apply(_factor_pair, axis=1)
        left = current.groupby("factor_pair")["coef"].agg(["size", "mean"])
        right = earlier.groupby("factor_pair")["interaction"].agg(
            ["size", "mean"])
        joined = left.join(right, lsuffix="_cross", rsuffix="_within",
                           how="inner")
        out.append(f"  factor pairs measured in both: {len(joined)}")
        if len(joined) >= 3:
            out.append(_corr(joined["mean_within"], joined["mean_cross"],
                             "mean I, earlier vs this"))
    return out


# --------------------------------------------------------------------------- #
# Exemplar search
# --------------------------------------------------------------------------- #

def find_exemplar_loci(
    inters_rna: pd.DataFrame,
    enh_inters: pd.DataFrame,
    *,
    interdep: pd.DataFrame | None = None,
    tf_pairs: pd.DataFrame | None = None,
    enh_mains: pd.DataFrame | None = None,
    measured_pairs: pd.DataFrame | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Rank enhancer pairs by the number of analysis levels that agree on them.

    Evidence (one point each): significant motif pair, significant enhancer pair,
    matching sign, within-enhancer support, significant mixed-class interaction.

    Args:
        inters_rna: Classified interaction terms on expression.
        enh_inters: Enhancer-level interaction coefficients.
        interdep: Motif interdependency table on accessibility.
        tf_pairs: Pairwise interactions from the within-enhancer run.
        enh_mains: Enhancer-level main effects, for context.
        measured_pairs: Measured pair table with interaction_beta.
        alpha: Significance threshold.

    Returns:
        One row per matched enhancer pair, sorted by evidence count and then by
        the enhancer-level coefficient magnitude.
    """
    pair_agg = aggregate_pairs(inters_rna, alpha=alpha)
    if pair_agg.empty or not _has(enh_inters, "coef"):
        return pd.DataFrame()

    candidates = pair_agg.copy()
    candidates["_key"] = _pair_key(candidates, "enhA", "enhB")

    enh = enh_inters.copy()
    enh["_key"] = _enh_pair_key(enh)
    enh = enh.rename(columns={"coef": "enh_coef", "qval": "enh_qval",
                              "distance_bp": "enh_distance_bp",
                              "main_a": "enh_main_a", "main_b": "enh_main_b"})
    keep = [c for c in ["_key", "enh_coef", "enh_qval", "enh_distance_bp",
                        "enh_main_a", "enh_main_b"] if c in enh.columns]
    candidates = candidates.merge(enh[keep].drop_duplicates("_key"),
                                  on="_key", how="inner")
    if candidates.empty:
        return candidates

    # Enhancer-level main effects give the reader the size of the two effects
    # the interaction is measured against.
    if _has(enh_mains, "coef", "full_a"):
        lookup = enh_mains.drop_duplicates("full_a").set_index("full_a")["coef"]
        candidates["enh_main_a"] = candidates["enhA"].map(lookup)
        candidates["enh_main_b"] = candidates["enhB"].map(lookup)

    if _has(measured_pairs, "interaction_beta"):
        screen = measured_pairs.copy()
        screen["_key"] = _enh_pair_key(screen)
        candidates = candidates.merge(
            screen[["_key", "interaction_beta"]].drop_duplicates("_key"),
            on="_key", how="left")

    candidates = _attach_within_support(candidates, inters_rna, tf_pairs, alpha)
    candidates = _attach_atac_support(candidates, interdep, alpha)

    sign_agree = (np.sign(candidates["motif_mean"])
                  == np.sign(candidates["enh_coef"]))
    candidates["sign_agree"] = sign_agree
    candidates["lead_sign_agree"] = (np.sign(candidates.get(
        "lead_coef", pd.Series(np.nan, index=candidates.index)))
        == np.sign(candidates["enh_coef"]))
    candidates["cross_significant"] = candidates["n_significant"] > 0
    candidates["enh_significant"] = (candidates["enh_qval"] < alpha) \
        if "enh_qval" in candidates.columns else False

    candidates["n_evidence"] = (
        candidates["cross_significant"].astype(int)
        + candidates["enh_significant"].astype(int)
        + candidates["sign_agree"].astype(int)
        + candidates["within_support"].astype(int)
        + candidates["atac_support"].astype(int)
    )

    candidates = candidates.sort_values(
        ["n_evidence", "enh_coef"],
        key=lambda s: s.abs() if s.name == "enh_coef" else s,
        ascending=[False, False])
    return candidates.reset_index(drop=True)


def _attach_within_support(candidates: pd.DataFrame, inters_rna: pd.DataFrame | None,
                           tf_pairs: pd.DataFrame | None, alpha: float) -> pd.DataFrame:
    """Add within-enhancer evidence from this run and the within-enhancer run."""
    out = candidates.copy()

    # Within-scope pairs of this run, indexed by the enhancer they sit in.
    if _has(inters_rna, "coef", "pair_scope", "enh_a", "gene"):
        within = inters_rna[inters_rna["pair_scope"] == "within"].dropna(
            subset=["coef"]).copy()
        within["_enh"] = within["gene"].astype(str) + "_" + \
            within["enh_a"].astype(str)
        within["_sig"] = (within["qval"] < alpha) \
            if "qval" in within.columns else False
        summary = within.groupby("_enh").agg(n=("coef", "size"),
                                             median=("coef", "median"),
                                             sig=("_sig", "sum"))
        for side, column in [("enhA", "a"), ("enhB", "b")]:
            out[f"within_run_n_{column}"] = out[side].map(summary["n"])
            out[f"within_run_median_{column}"] = out[side].map(summary["median"])
            out[f"within_run_sig_{column}"] = out[side].map(summary["sig"])
    else:
        for column in ("a", "b"):
            out[f"within_run_n_{column}"] = np.nan
            out[f"within_run_median_{column}"] = np.nan
            out[f"within_run_sig_{column}"] = np.nan

    # The earlier within-enhancer run, matched on the enhancer identifier and,
    # separately, on the factor pair that leads the cross-enhancer signal.
    out["earlier_n_a"] = np.nan
    out["earlier_n_b"] = np.nan
    out["earlier_lead_pair_I"] = np.nan
    out["earlier_lead_pair_where"] = ""
    if _has(tf_pairs, "interaction", "enhancer", "motif_a", "motif_b"):
        earlier = tf_pairs.dropna(subset=["interaction"]).copy()
        earlier["factor_pair"] = earlier.apply(_factor_pair, axis=1)
        counts = earlier.groupby("enhancer")["interaction"].size()
        out["earlier_n_a"] = out["enhA"].map(counts)
        out["earlier_n_b"] = out["enhB"].map(counts)

        if {"lead_motif_a", "lead_motif_b"}.issubset(out.columns):
            lead_pairs = out.apply(
                lambda r: _factor_pair(r, "lead_motif_a", "lead_motif_b"),
                axis=1)
            for index, factor_pair in lead_pairs.items():
                enhancers = {out.at[index, "enhA"], out.at[index, "enhB"]}
                hit = earlier[(earlier["factor_pair"] == factor_pair)
                              & (earlier["enhancer"].isin(enhancers))]
                if not hit.empty:
                    strongest = hit.loc[hit["interaction"].abs().idxmax()]
                    out.at[index, "earlier_lead_pair_I"] = \
                        float(strongest["interaction"])
                    out.at[index, "earlier_lead_pair_where"] = \
                        str(strongest["enhancer"])

    within_counts = out[["within_run_n_a", "within_run_n_b",
                         "earlier_n_a", "earlier_n_b"]].fillna(0)
    out["within_support"] = (within_counts.sum(axis=1) > 0) | \
        out["earlier_lead_pair_I"].notna()
    return out


def _attach_atac_support(candidates: pd.DataFrame, interdep: pd.DataFrame | None,
                         alpha: float) -> pd.DataFrame:
    """Add mixed-class accessibility evidence for the same enhancer pair."""
    out = candidates.copy()
    out["atac_mixed_n"] = 0
    out["atac_mixed_sig"] = 0
    out["atac_mixed_median"] = np.nan
    out["atac_support"] = False

    if not _has(interdep, "delta", "level", "cis_class", "gene"):
        return out

    mixed = interdep[(interdep["level"] == "interaction")
                     & (interdep["cis_class"] == "mixed")].copy()
    if mixed.empty or not {"enh_a", "enh_b"}.issubset(mixed.columns):
        return out

    mixed["_key"] = mixed.apply(
        lambda r: "||".join(sorted([f"{r['gene']}_{r['enh_a']}",
                                    f"{r['gene']}_{r['enh_b']}"])), axis=1)
    mixed["_sig"] = (mixed["qval"] < alpha) if "qval" in mixed.columns else False
    summary = mixed.groupby("_key").agg(n=("delta", "size"),
                                        median=("delta", "median"),
                                        sig=("_sig", "sum"))
    out["atac_mixed_n"] = out["_key"].map(summary["n"]).fillna(0).astype(int)
    out["atac_mixed_sig"] = out["_key"].map(summary["sig"]).fillna(0).astype(int)
    out["atac_mixed_median"] = out["_key"].map(summary["median"])
    out["atac_support"] = out["atac_mixed_sig"] > 0
    return out


def _format_exemplar(row: pd.Series, rank: int, alpha: float) -> list[str]:
    """Format one candidate enhancer pair as a text block."""
    def value(name: str, default: Any = np.nan) -> Any:
        return row[name] if name in row.index else default

    def number(name: str, fmt: str = "{:+.4f}") -> str:
        raw = value(name)
        return fmt.format(raw) if isinstance(raw, (int, float, np.number)) \
            and np.isfinite(raw) else "n/a"

    def qvalue(name: str) -> str:
        raw = value(name)
        if not isinstance(raw, (int, float, np.number)) or not np.isfinite(raw):
            return "n/a"
        if raw <= 0:
            return "<1e-300"   # underflow in the adjusted p-value
        return f"{raw:.1e}" if raw < 1e-3 else f"{raw:.4f}"

    gene = value("gene", "?")
    pair = str(value("enh_pair", "?")).replace("_x_", " x ")
    out = ["", f"  [{rank}] {gene}   {pair}"
               f"   evidence {int(value('n_evidence', 0))}/5"]

    out.append(f"      cross-enhancer motifs   n pairs="
               f"{int(value('n_motif_pairs', 0))}  "
               f"mean beta_ij={number('motif_mean')}  "
               f"most negative={number('motif_strongest')}  "
               f"significant={int(value('n_significant', 0))}")
    if "lead_motif_a" in row.index:
        out.append(f"        lead motif pair       "
                   f"{value('lead_motif_a', '?')} ({value('lead_enh_a', '?')})"
                   f"  x  {value('lead_motif_b', '?')} "
                   f"({value('lead_enh_b', '?')})")
        out.append(f"                              "
                   f"beta_ij={number('lead_coef')}  "
                   f"q={qvalue('lead_qval')}  "
                   f"separation={number('lead_distance_bp', '{:,.0f}')} bp")

    out.append(f"      enhancer knockout       "
               f"beta_ij={number('enh_coef')}  "
               f"q={qvalue('enh_qval')}  "
               f"distance={number('enh_distance_bp', '{:,.0f}')} bp")
    if "enh_main_a" in row.index:
        out.append(f"        component mains       "
                   f"beta_A={number('enh_main_a')}  "
                   f"beta_B={number('enh_main_b')}")
    if "interaction_beta" in row.index:
        out.append(f"      CRISPRi measured        "
                   f"interaction={number('interaction_beta')}")

    out.append(f"      accessibility, mixed    "
               f"n={int(value('atac_mixed_n', 0))}  "
               f"median={number('atac_mixed_median')}  "
               f"significant={int(value('atac_mixed_sig', 0))}")

    def count(name: str) -> int:
        raw = value(name)
        return int(raw) if isinstance(raw, (int, float, np.number)) \
            and np.isfinite(raw) else 0

    within_lines = []
    for side, column, enh_col in [("A", "a", "enhA"), ("B", "b", "enhB")]:
        n_here = count(f"within_run_n_{column}")
        n_earlier = count(f"earlier_n_{column}")
        if not (n_here or n_earlier):
            continue
        parts = []
        if n_here:
            parts.append(f"{n_here} within-scope pair(s) here, median "
                         f"{number(f'within_run_median_{column}')}")
        if n_earlier:
            parts.append(f"{n_earlier} pair(s) in the earlier run")
        within_lines.append(f"{side} ({value(enh_col, '?')}): "
                            + "; ".join(parts))
    if within_lines:
        out.append(f"      within-enhancer         {within_lines[0]}")
        for extra in within_lines[1:]:
            out.append(f"                              {extra}")
    else:
        out.append("      within-enhancer         no support")
    if isinstance(value("earlier_lead_pair_I"), (int, float, np.number)) \
            and np.isfinite(value("earlier_lead_pair_I")):
        out.append(f"        same factor pair      measured in "
                   f"{value('earlier_lead_pair_where', '?')} at the "
                   f"within-enhancer level, I="
                   f"{number('earlier_lead_pair_I')}")

    direction = "super-additive" if value("enh_coef", 0) < 0 else "sub-additive"
    agreement = "agree" if bool(value("sign_agree", False)) else "disagree"
    out.append(f"      summary                 enhancer level is {direction}; "
               f"motif mean and enhancer coefficient {agreement} in sign")
    return out


def _section_exemplars(inters_rna: pd.DataFrame | None, enh_inters: pd.DataFrame | None, interdep: pd.DataFrame | None,
                       tf_pairs: pd.DataFrame | None, enh_mains: pd.DataFrame | None, measured_pairs: pd.DataFrame | None,
                       alpha: float, exemplar_n: int) -> list[str]:
    """Section 12: exemplar loci."""
    out = ["", _RULE, "12.  EXEMPLAR LOCI", _RULE,
           "  Enhancer pairs at which several levels of the analysis speak to",
           "  the same object. Ranked by the number of independent lines of",
           "  evidence, then by the size of the enhancer-level coefficient.",
           "  Selection is illustrative; these pairs were chosen because they",
           "  agree, so their agreement is not evidence about the population."]

    if not _has(inters_rna, "coef") or not _has(enh_inters, "coef"):
        return out + _missing("inters_rna or enh_inters")

    candidates = find_exemplar_loci(
        inters_rna, enh_inters, interdep=interdep, tf_pairs=tf_pairs,
        enh_mains=enh_mains, measured_pairs=measured_pairs, alpha=alpha)
    if candidates.empty:
        return out + ["", "  no enhancer pair is present at both levels"]

    out.append("")
    out.append(f"  matched enhancer pairs: {len(candidates)}")
    distribution = candidates["n_evidence"].value_counts().sort_index(
        ascending=False)
    out.append("  evidence count distribution: "
               + ", ".join(f"{int(k)}/5: {int(v)}"
                           for k, v in distribution.items()))
    out.append(f"  sign agreement between motif mean and enhancer coefficient: "
               f"{candidates['sign_agree'].mean():.1%} of {len(candidates)}")

    for rank, (_, row) in enumerate(candidates.head(exemplar_n).iterrows(),
                                    start=1):
        out += _format_exemplar(row, rank, alpha)

    out += ["", "  Counter-examples, the pairs where the two levels disagree "
                "most:"]
    disagree = candidates[~candidates["sign_agree"].fillna(False)]
    if disagree.empty:
        out.append("      (none disagree in sign)")
    else:
        ranked = disagree.reindex(
            disagree["enh_coef"].abs().sort_values(ascending=False).index)
        for _, row in ranked.head(3).iterrows():
            out.append(f"      {str(row.get('enh_pair', '?')):<36}"
                       f" motif mean={row.get('motif_mean', np.nan):+.4f}"
                       f"  enhancer beta_ij={row.get('enh_coef', np.nan):+.4f}")
    return out


# --------------------------------------------------------------------------- #
# Flags
# --------------------------------------------------------------------------- #

def _section_flags(inters_rna: pd.DataFrame | None, beta_rna: pd.DataFrame | None, interdep: pd.DataFrame | None,
                   enh_inters: pd.DataFrame | None, tf_pairs: pd.DataFrame | None,
                   glm_results_rna: dict | None, glm_results_atac: dict | None,
                   controls: pd.DataFrame | None, alpha: float) -> list[str]:
    """Section 13: automatic flags for patterns to check before interpretation."""
    out = ["", _RULE, "13.  FLAGS", _RULE]
    flags = []

    if controls is None or (isinstance(controls, pd.DataFrame)
                            and controls.empty):
        flags.append("No cross-enhancer specificity control. Motif effects at "
                     "this level cannot be separated from generic sequence "
                     "perturbation; specificity was established only in the "
                     "within-enhancer run, in a differently anchored window.")

    for label, results in [("RNA", glm_results_rna), ("ATAC", glm_results_atac)]:
        if not results:
            continue
        failed = [k for k, r in results.items() if not r.get("converged")]
        deficient = [k for k, r in results.items()
                     if (r.get("identifiability") or {}).get("full_rank") is False]
        if failed:
            flags.append(f"{len(failed)} {label} fit(s) did not converge, so "
                         f"those genes contribute no terms.")
        if deficient:
            flags.append(f"{len(deficient)} {label} design(s) are rank "
                         f"deficient. Coefficients are still returned for "
                         f"unestimable terms and must not be read.")

    if _has(inters_rna, "coef", "pair_scope"):
        counts = inters_rna["pair_scope"].value_counts()
        n_within = int(counts.get("within", 0))
        n_across = int(counts.get("across", 0))
        if min(n_within, n_across) < 20:
            flags.append(f"Scope groups are unbalanced or small "
                         f"(within={n_within}, across={n_across}). The scope "
                         f"contrast is the headline comparison and rests on "
                         f"these counts.")
        values = inters_rna["coef"].dropna()
        if len(values) and min((values < 0).mean(), (values > 0).mean()) < 0.10:
            direction = "sub-additive" if (values > 0).mean() > 0.9 \
                else "super-additive"
            flags.append(f"Interactions are almost entirely {direction}. A "
                         f"one-sided result often indicates a systematic "
                         f"cause rather than a biological one.")
        if "qval" in inters_rna.columns:
            n_sig = int((inters_rna["qval"] < alpha).sum())
            if n_sig < 10:
                flags.append(f"Only {n_sig} motif pairs survive correction. "
                             f"Limit claims to the aggregate distributions.")
        if "distance_bp" in inters_rna.columns:
            flags.append("Distance and scope are collinear by construction, "
                         "since across-enhancer pairs are separated by orders "
                         "of magnitude more sequence. Report the within-scope "
                         "correlations, not the pooled one.")

    if beta_rna is not None and _has(inters_rna, "pval"):
        mains = _split_mains(beta_rna)
        if not mains.empty and "qval" in inters_rna.columns:
            refit = _bh(inters_rna["pval"])
            before = int((inters_rna["qval"] < alpha).sum())
            after = int((refit < alpha).sum())
            if before != after:
                flags.append(f"Correction family: {before} interactions pass "
                             f"with the supplied q-values but {after} pass "
                             f"when BH is applied to the interaction terms "
                             f"alone. Pooling main effects with interactions "
                             f"is anti-conservative for the interactions.")

    if _has(interdep, "cis_class", "level"):
        pairs = interdep[interdep["level"] == "interaction"]
        n_mixed = int((pairs["cis_class"] == "mixed").sum())
        if n_mixed < 20:
            flags.append(f"The mixed class holds {n_mixed} cells. This is the "
                         f"most direct evidence the design produces for "
                         f"communication between elements, and it is thin.")

    for name, table in [("enh_inters", enh_inters), ("tf_pairs", tf_pairs)]:
        if table is None or (isinstance(table, pd.DataFrame) and table.empty):
            flags.append(f"{name} was not supplied, so the corresponding "
                         f"cross-level comparison is untested.")

    flags.append("Replicates are dinucleotide shuffles and are technical. "
                 "Every p-value in this document describes robustness to the "
                 "shuffle procedure, not biological variability.")

    for index, flag in enumerate(flags, start=1):
        out.append(f"  [{index}] {flag}")
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def build_cross_summary(
    *,
    inters_rna: pd.DataFrame | None = None,
    beta_rna: pd.DataFrame | None = None,
    inters_atac: pd.DataFrame | None = None,
    beta_atac: pd.DataFrame | None = None,
    interdep: pd.DataFrame | None = None,
    pair_agg: pd.DataFrame | None = None,
    scope: dict | None = None,
    enh_inters: pd.DataFrame | None = None,
    enh_mains: pd.DataFrame | None = None,
    measured_pairs: pd.DataFrame | None = None,
    tf_pairs: pd.DataFrame | None = None,
    tf_motifs: pd.DataFrame | None = None,
    seqlets: pd.DataFrame | None = None,
    controls: pd.DataFrame | None = None,
    glm_results_rna: dict | None = None,
    glm_results_atac: dict | None = None,
    alpha: float = 0.05,
    top_n: int = 10,
    exemplar_n: int = 3,
    title: str = "K562 cross-enhancer TF motif cooperativity",
) -> str:
    """Assemble the cross-enhancer text summary from whichever tables are supplied.

    Args:
        inters_rna: Classified interaction terms on expression, ideally with
            distance_bp.
        beta_rna: Full expression coefficient table.
        inters_atac: Classified interaction terms on accessibility.
        beta_atac: Full accessibility coefficient table.
        interdep: Output of glm.build_motif_interdependency.
        pair_agg: Enhancer-pair aggregation; recomputed when None.
        scope: Output of glm.summarise_scope_comparison, echoed for reference.
        enh_inters: Enhancer-level interaction coefficients.
        enh_mains: Enhancer-level main effects.
        measured_pairs: Measured pair table with interaction_beta.
        tf_pairs: Pairwise interactions from the within-enhancer run.
        tf_motifs: Motif effects from the within-enhancer run.
        seqlets: Selected seqlet table, for coverage.
        controls: Control window results, if available.
        glm_results_rna: Output of glm.fit_all_genes.
        glm_results_atac: Output of glm.fit_all_loci.
        alpha: Significance threshold.
        top_n: Number of named extremes per direction.
        exemplar_n: Number of exemplar loci described.
        title: Document title.

    Returns:
        The summary text.
    """
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [_RULE, title, f"generated {stamp}",
             f"alpha={alpha}   top_n={top_n}", _RULE]

    supplied = dict(
        inters_rna=inters_rna, beta_rna=beta_rna, inters_atac=inters_atac,
        beta_atac=beta_atac, interdep=interdep, pair_agg=pair_agg,
        enh_inters=enh_inters, enh_mains=enh_mains,
        measured_pairs=measured_pairs, tf_pairs=tf_pairs,
        tf_motifs=tf_motifs, seqlets=seqlets, controls=controls,
    )

    lines += ["", _THIN, "1.  INPUTS SUPPLIED", _THIN]
    for name, table in supplied.items():
        if table is None:
            lines.append(f"  {name:<20} MISSING")
        elif isinstance(table, pd.DataFrame):
            lines.append(f"  {name:<20} {len(table):>6} rows x "
                         f"{len(table.columns):>3} cols")
    for name, results in [("glm_results_rna", glm_results_rna),
                          ("glm_results_atac", glm_results_atac)]:
        lines.append(f"  {name:<20} "
                     + (f"{len(results):>6} fits" if results else "MISSING"))
    lines.append(f"  {'scope summary':<20} "
                 + ("supplied" if scope else "MISSING"))

    lines += ["", _THIN, "2.  READING THIS DOCUMENT", _THIN,
              "  Effects are log fold changes against the unperturbed",
              "  sequence, so scrambling a functional motif gives a negative",
              "  value. The interaction is the deviation from a multiplicative",
              "  null, beta_ij = delta_ij - delta_i - delta_j, therefore",
              "  beta_ij < 0 is SUPER-ADDITIVE and beta_ij > 0 SUB-ADDITIVE.",
              "",
              "  Unit of perturbation: one seqlet, labelled by the enhancer",
              "  that contains it, for example enh1|GATA1. A pair is WITHIN",
              "  scope when both seqlets sit in one enhancer and ACROSS scope",
              "  when they sit in different enhancers of the same gene.",
              "",
              "  Both scopes were measured in the same 1 Mb window, so the",
              "  comparison between them is controlled for genomic context.",
              "  Neither is numerically comparable to the earlier",
              "  within-enhancer run, which used a differently anchored",
              "  window; section 11 makes that comparison explicitly and",
              "  descriptively.",
              "",
              "  Expression is predicted RNA at the target gene. Accessibility",
              "  is predicted ATAC at each enhancer of the gene separately, so",
              "  a motif has an effect on its own element and on the others."]

    lines += _section_design(inters_rna, beta_rna, glm_results_rna,
                             glm_results_atac, seqlets)
    lines += _section_main_effects(beta_rna, inters_rna, alpha, top_n)
    lines += _section_scope(inters_rna, beta_rna, scope, alpha, top_n)
    lines += _section_grammar(inters_rna, alpha, top_n)
    lines += _section_distance(inters_rna)
    lines += _section_interdependency(interdep, alpha, top_n)
    lines += _section_readouts(inters_rna, inters_atac, interdep, alpha)
    lines += _section_cross_level(inters_rna, pair_agg, enh_inters, enh_mains,
                                  measured_pairs, alpha)
    lines += _section_within_run(inters_rna, interdep, tf_pairs, alpha)
    lines += _section_exemplars(inters_rna, enh_inters, interdep, tf_pairs,
                                enh_mains, measured_pairs, alpha, exemplar_n)
    lines += _section_flags(inters_rna, beta_rna, interdep, enh_inters,
                            tf_pairs, glm_results_rna, glm_results_atac,
                            controls, alpha)

    lines += ["", _RULE, "END", _RULE]
    return "\n".join(lines)


def write_cross_summary(path: str, **kwargs: Any) -> str:
    """Build the cross-enhancer summary and write it to a file.

    Args:
        path: Output file.
        **kwargs: Passed to build_cross_summary.

    Returns:
        The summary text.
    """
    text = build_cross_summary(**kwargs)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    print(f"Wrote {path} ({len(text.splitlines())} lines, "
          f"{len(text) / 1024:.1f} KB)")
    return text