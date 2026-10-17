"""Log-link GLM for factorial in silico perturbation designs.

Model: log E[y] = b0 + sum_i b_i 1[i scrambled] + sum_{i<j} b_ij 1[i and j scrambled],
so b_ij is the deviation from a multiplicative null (negative: super-additive).
The perturbation unit is a whole enhancer ("GENE_enh1"), a seqlet ("GATA1") or
an enhancer-qualified seqlet ("GENE_enh1|ATF4").
"""

from __future__ import annotations

import ast
import itertools
import warnings
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial as SM_NB


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def parse_sequence_column(value: Any) -> list:
    """Convert a list-like cell (list, array or its string form) into a Python list.

    Args:
        value: Cell value.

    Returns:
        The value as a list; an empty list when it cannot be parsed.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, str):
        if value.startswith("array("):
            return []
        try:
            parsed = ast.literal_eval(value)
            return list(parsed) if isinstance(parsed, (list, tuple)) else []
        except (ValueError, SyntaxError):
            return []
    return []


def short_label(unit: str) -> str:
    """Strip the gene prefix from a unit name ("GENE_enh1|ATF4" -> "enh1|ATF4").

    Args:
        unit: Fully qualified unit name.

    Returns:
        The label used in coefficient tables.
    """
    return unit.split("_", 1)[-1] if "_" in unit else unit


# --------------------------------------------------------------------------- #
# Design matrix
# --------------------------------------------------------------------------- #

def build_design_matrix(
    scr_lists: list[list[str]],
    units: list[str],
    max_order: int,
) -> tuple[np.ndarray, list[str]]:
    """Build an intercept, main-effect and interaction design matrix.

    A term for a set of units is 1 when every unit in the set is scrambled.

    Args:
        scr_lists: Scrambled units per observation.
        units: Ordered perturbable units of the gene.
        max_order: Highest interaction order (1 main effects, 2 pairs, ...).

    Returns:
        Tuple of (design matrix with intercept column, term names).
    """
    n_units = len(units)
    max_order = min(max_order, n_units)

    term_groups = []
    for order in range(1, max_order + 1):
        for combo in itertools.combinations(range(n_units), order):
            members = [units[i] for i in combo]
            label = ":".join(short_label(u) for u in members)
            term_groups.append((label, frozenset(members)))

    scr_sets = [frozenset(sl) for sl in scr_lists]
    design = np.zeros((len(scr_sets), len(term_groups)))
    for j, (_, term_set) in enumerate(term_groups):
        for i, scr_set in enumerate(scr_sets):
            design[i, j] = 1.0 if term_set <= scr_set else 0.0

    design = np.column_stack([np.ones(len(scr_sets)), design])
    return design, ["intercept"] + [t[0] for t in term_groups]


def check_identifiability(design: np.ndarray, term_names: list[str]) -> dict:
    """Check whether every term of a (possibly reduced) design is estimable.

    Args:
        design: Design matrix including the intercept.
        term_names: Matching term names.

    Returns:
        Dictionary with the matrix rank, the rank deficiency and the terms whose
        columns are constant.
    """
    rank = int(np.linalg.matrix_rank(design))
    n_cols = design.shape[1]
    constant = [name for name, col in zip(term_names, design.T)
                if np.ptp(col) == 0 and name != "intercept"]
    return dict(rank=rank, n_terms=n_cols, deficiency=n_cols - rank,
                constant_terms=constant, full_rank=rank == n_cols)


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #

def _fit_glm(y: np.ndarray, design: np.ndarray, term_names: list[str], family: str,
             round_nb: bool) -> pd.DataFrame | None:
    """Fit one Gamma or negative binomial GLM and return a tidy coefficient table, or None on failure."""
    try:
        if family == "nb":
            y_fit = np.round(y).astype(int) if round_nb else y.astype(int)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = SM_NB(y_fit, design).fit(method="bfgs", disp=False,
                                                  maxiter=200)
        elif family == "gamma":
            y_fit = np.clip(y, 1e-6, None)   # gamma requires strictly positive
            result = sm.GLM(y_fit, design,
                            family=sm.families.Gamma(
                                link=sm.families.links.Log())).fit(disp=False)
        else:
            raise ValueError(f"Unknown family '{family}'. Use 'nb' or 'gamma'.")
    except Exception as error:
        warnings.warn(f"GLM fit failed: {error}")
        return None

    ci = np.asarray(result.conf_int())
    params = np.asarray(result.params)

    # The negative binomial appends a dispersion parameter, so the name list
    # has to grow to match or the DataFrame construction fails.
    local_terms = list(term_names)
    if len(params) == len(local_terms) + 1:
        local_terms.append("alpha")
    elif len(params) != len(local_terms):
        warnings.warn(f"Parameter length mismatch: expected {len(local_terms)}, "
                      f"got {len(params)}")
        return None

    table = pd.DataFrame(dict(
        term=local_terms,
        coef=params,
        se=np.asarray(result.bse),
        ci_lo=ci[:, 0],
        ci_hi=ci[:, 1],
        pval=np.asarray(result.pvalues),
        family=family,
    ))
    # Carried through so the model comparison panel has something to plot.
    table.attrs["aic"] = float(getattr(result, "aic", np.nan))
    table.attrs["deviance"] = float(getattr(result, "deviance", np.nan))
    return table


def fit_interaction_glm(
    df_raw: pd.DataFrame,
    gene: str,
    family: str = "gamma",
    max_order: int = 2,
    response_col: str = "RNA_counts",
    units_col: str | None = None,
    units: list[str] | None = None,
    scrambled_col: str = "scrambled_seqlets",
    name_col: str = "name",
    round_nb: bool = True,
    verbose: bool = False,
    rna_col: str | None = None,
) -> dict:
    """Fit the interaction GLM of one gene on the unaggregated scramble table.

    Units are taken from, in order: units, the baseline row of units_col, the
    baseline row of unscrambled_seqlets, or the baseline row of target_enhancer.

    Args:
        df_raw: Scramble results, one row per shuffle replicate.
        gene: Value of name_col to fit.
        family: "gamma" or "nb".
        max_order: Highest interaction order to include.
        response_col: Scalar response column.
        units_col: Column whose baseline value lists the perturbable units.
        units: Explicit unit list, overriding units_col.
        scrambled_col: Column with the scrambled units per row.
        name_col: Column identifying the gene.
        round_nb: Round the response to integers for the negative binomial.
        verbose: Print the coefficient table.
        rna_col: Deprecated alias for response_col.

    Returns:
        Dictionary with coef_table, units, n_obs, an identifiability report and a
        convergence flag.
    """
    if rna_col is not None:
        response_col = rna_col   # backwards compatibility with older calls

    grp = df_raw[df_raw[name_col] == gene].copy().reset_index(drop=True)
    if grp.empty:
        raise ValueError(f"Gene '{gene}' not found in column '{name_col}'.")

    grp["_scr"] = grp[scrambled_col].apply(parse_sequence_column)
    baseline_rows = grp[grp["n_scrambled"].astype(int) == 0]
    if baseline_rows.empty:
        raise ValueError(f"{gene}: no baseline row (n_scrambled == 0).")
    base = baseline_rows.iloc[0]

    if units is not None:
        resolved_units = list(units)
        source = "explicit"
    else:
        candidates = ([units_col] if units_col else
                      ["unscrambled_seqlets", "target_enhancer"])
        resolved_units, source = None, None
        for column in candidates:
            if column in grp.columns:
                candidate = parse_sequence_column(base[column])
                if candidate:
                    resolved_units, source = list(candidate), column
                    break
        if not resolved_units:
            raise ValueError(
                f"{gene}: could not resolve perturbation units. Tried "
                f"{candidates}. Pass `units` or `units_col` explicitly."
            )

    # The units must actually be the objects that appear in the scrambled
    # lists, otherwise every indicator is zero and only the intercept fits.
    observed = {u for lst in grp["_scr"] for u in lst}
    unmatched = observed - set(resolved_units)
    if unmatched:
        raise ValueError(
            f"{gene}: {len(unmatched)} scrambled unit(s) are absent from the "
            f"unit list resolved from '{source}', for example "
            f"{sorted(unmatched)[:3]}. The design would be all zeros. "
            f"Check `units_col`."
        )

    y = grp[response_col].astype(float).values
    valid = np.isfinite(y)
    if valid.sum() < len(resolved_units) + 2:
        warnings.warn(f"{gene}: too few valid observations ({int(valid.sum())}).")
        return dict(gene=gene, converged=False, coef_table=None,
                    units=resolved_units, enhancers=resolved_units,
                    n_obs=int(valid.sum()), family=family, max_order=max_order,
                    identifiability=None, response=response_col)

    design, term_names = build_design_matrix(
        grp.loc[valid, "_scr"].tolist(), resolved_units, max_order)
    identifiability = check_identifiability(design, term_names)
    if not identifiability["full_rank"]:
        warnings.warn(
            f"{gene}: design is rank deficient by "
            f"{identifiability['deficiency']} of {identifiability['n_terms']} "
            f"terms. Coefficients for unestimable terms are not meaningful."
        )

    coef_table = _fit_glm(y[valid], design, term_names, family, round_nb)

    if verbose and coef_table is not None:
        print(f"\n{'-'*60}\n  {gene} | {family} | order {max_order} | "
              f"n={int(valid.sum())} | units from '{source}' "
              f"({len(resolved_units)})\n{'-'*60}")
        print(coef_table.to_string(index=False, float_format="{:.4f}".format))

    return dict(
        gene=gene,
        family=family,
        max_order=max_order,
        coef_table=coef_table,
        units=resolved_units,
        enhancers=resolved_units,        # historical alias
        unit_source=source,
        n_obs=int(valid.sum()),
        converged=coef_table is not None,
        identifiability=identifiability,
        response=response_col,
        aic=coef_table.attrs.get("aic") if coef_table is not None else None,
        deviance=coef_table.attrs.get("deviance") if coef_table is not None else None,
    )


def fit_all_genes(
    df_raw: pd.DataFrame,
    genes: list[str] | None = None,
    name_col: str = "name",
    **kwargs: Any,
) -> dict[str, dict]:
    """Fit fit_interaction_glm for every gene.

    Args:
        df_raw: Scramble results, one row per shuffle replicate.
        genes: Genes to fit; all values of name_col when None.
        name_col: Column identifying the gene.
        **kwargs: Passed to fit_interaction_glm.

    Returns:
        Mapping of gene to its result.
    """
    if genes is None:
        genes = list(df_raw[name_col].unique())

    results = {}
    for gene in genes:
        try:
            results[gene] = fit_interaction_glm(df_raw, gene, name_col=name_col,
                                                **kwargs)
        except Exception as error:
            warnings.warn(f"Failed for {gene}: {error}")
    return results


def fit_all_loci(
    long_df: pd.DataFrame,
    response_col: str = "ATAC_counts",
    gene_col: str = "name",
    locus_col: str = "locus",
    **kwargs: Any,
) -> dict[str, dict]:
    """Fit one GLM per measured locus for the per-enhancer accessibility readout.

    Args:
        long_df: Output of enhancer_analysis.expand_locus_values.
        response_col: Scalar response column.
        gene_col: Column identifying the gene.
        locus_col: Column identifying the measured locus.
        **kwargs: Passed to fit_interaction_glm.

    Returns:
        Mapping of "gene@locus" to its result, each carrying gene and locus keys.
    """
    results = {}
    for (gene, locus), block in long_df.groupby([gene_col, locus_col]):
        key = f"{gene}@{locus}"
        try:
            # The helper selects on name_col, so present the locus as the name.
            block = block.copy()
            block["_fit_key"] = key
            result = fit_interaction_glm(block, key, name_col="_fit_key",
                                         response_col=response_col, **kwargs)
            result["gene"] = gene
            result["locus"] = locus
            results[key] = result
        except Exception as error:
            warnings.warn(f"Failed for {key}: {error}")
    return results


# --------------------------------------------------------------------------- #
# Coefficient collection
# --------------------------------------------------------------------------- #

def collect_terms(
    results: dict[str, dict],
    interactions_only: bool = False,
) -> pd.DataFrame:
    """Stack the fitted coefficient tables into one frame.

    Args:
        results: Output of fit_all_genes or fit_all_loci.
        interactions_only: Keep only interaction terms.

    Returns:
        One row per term with gene, and locus and response where defined.
    """
    rows = []
    for key, result in results.items():
        if not result.get("converged") or result.get("coef_table") is None:
            continue
        table = result["coef_table"].copy()
        table["gene"] = result.get("gene", key)
        if "locus" in result:
            table["locus"] = result["locus"]
        table["response"] = result.get("response")
        table["order"] = table["term"].str.count(":") + 1
        table.loc[table["term"].isin(["intercept", "alpha"]), "order"] = 0
        rows.append(table)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out[~out["term"].isin(["intercept", "alpha"])]
    if interactions_only:
        out = out[out["term"].str.contains(":")]
    return out.reset_index(drop=True)


def classify_interaction_terms(
    inters: pd.DataFrame,
    term_col: str = "term",
    gene_col: str = "gene",
    unit_separator: str = "|",
) -> pd.DataFrame:
    """Split interaction terms into their two units and label each pair within or across enhancer.

    Args:
        inters: Interaction terms from collect_terms.
        term_col: Column with the colon-joined term.
        gene_col: Column with the gene.
        unit_separator: Separator between enhancer and motif in a unit label.

    Returns:
        Copy of inters with unit, enhancer and motif per side, pair_scope
        ("within" or "across") and fully qualified full_a and full_b.
    """
    if inters.empty:
        return inters.copy()

    # Match only single colons, via negative lookbehind and lookahead, so that
    # dimer motif labels such as MAFG::NRF1 are neither treated as an
    # interaction separator nor split apart.
    single_colon = r"(?<!:):(?!:)"
    out = inters[inters[term_col].str.contains(single_colon, regex=True)].copy()
    parts = out[term_col].str.split(single_colon, n=1, expand=True, regex=True)
    out["unit_a"], out["unit_b"] = parts[0], parts[1]

    for side in ("a", "b"):
        unit = out[f"unit_{side}"]
        has_sep = unit.str.contains(unit_separator, regex=False)
        out[f"enh_{side}"] = np.where(has_sep,
                                      unit.str.split(unit_separator).str[0],
                                      unit)
        out[f"motif_{side}"] = np.where(has_sep,
                                        unit.str.split(unit_separator).str[-1],
                                        "")
        out[f"full_{side}"] = out[gene_col] + "_" + unit

    out["pair_scope"] = np.where(out["enh_a"] == out["enh_b"], "within", "across")
    # Enhancer pair key, order independent, for joining to the enhancer-level
    # interaction coefficients.
    ordered = out.apply(
        lambda r: tuple(sorted([f"{r[gene_col]}_{r['enh_a']}",
                                f"{r[gene_col]}_{r['enh_b']}"])), axis=1)
    out["enhA"] = [p[0] for p in ordered]
    out["enhB"] = [p[1] for p in ordered]
    out["enh_pair"] = out["enhA"] + "_x_" + out["enhB"]
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Cross-enhancer helpers
# --------------------------------------------------------------------------- #

def classify_main_terms(
    terms: pd.DataFrame,
    term_col: str = "term",
    gene_col: str = "gene",
    unit_separator: str = "|",
) -> pd.DataFrame:
    """Split main-effect terms into enhancer and motif.

    Args:
        terms: Coefficient table from collect_terms.
        term_col: Column with the term name.
        gene_col: Column with the gene.
        unit_separator: Separator between enhancer and motif in a unit label.

    Returns:
        Main effects with enh, motif and full_unit, plus is_cis when a locus
        column is present.
    """
    if terms.empty:
        return terms.copy()

    single_colon = r"(?<!:):(?!:)"
    out = terms[~terms[term_col].str.contains(single_colon, regex=True)].copy()
    out = out[~out[term_col].isin(["intercept", "alpha"])]
    if out.empty:
        return out

    has_sep = out[term_col].str.contains(unit_separator, regex=False)
    out["enh"] = np.where(has_sep,
                          out[term_col].str.split(unit_separator).str[0],
                          out[term_col])
    out["motif"] = np.where(has_sep,
                            out[term_col].str.split(unit_separator).str[-1], "")
    out["full_unit"] = out[gene_col] + "_" + out[term_col]
    out["full_enh"] = out[gene_col] + "_" + out["enh"]

    if "locus" in out.columns:
        # A cis row is one where the perturbed motif lies in the locus being
        # measured; everything else is a trans effect between enhancers.
        out["is_cis"] = out["full_enh"] == out["locus"]
    return out.reset_index(drop=True)


def summarise_scope_comparison(
    inters: pd.DataFrame,
    value_col: str = "coef",
    scope_col: str = "pair_scope",
    qval_col: str = "qval",
    alpha: float = 0.05,
) -> dict:
    """Compare interaction coefficients of within- and across-enhancer pairs.

    Args:
        inters: Classified interaction terms.
        value_col: Interaction coefficient column.
        scope_col: Column with "within" or "across".
        qval_col: Adjusted p-value column.
        alpha: Significance threshold.

    Returns:
        Per-scope counts, medians, direction splits and significant counts, and a
        Mann-Whitney test between the scopes.
    """
    from scipy import stats as _stats

    out = {}
    groups = {}
    for scope in ("within", "across"):
        block = inters[inters[scope_col] == scope].dropna(subset=[value_col])
        values = block[value_col].to_numpy()
        groups[scope] = values
        entry = dict(
            n=len(values),
            median=float(np.median(values)) if len(values) else np.nan,
            mean=float(np.mean(values)) if len(values) else np.nan,
            n_super=int((values < 0).sum()),
            n_sub=int((values > 0).sum()),
        )
        if qval_col in block.columns:
            hit = block[qval_col] < alpha
            entry["n_significant"] = int(hit.sum())
            entry["n_sig_super"] = int((hit & (block[value_col] < 0)).sum())
            entry["n_sig_sub"] = int((hit & (block[value_col] > 0)).sum())
        out[scope] = entry

    if len(groups["within"]) > 2 and len(groups["across"]) > 2:
        stat, pval = _stats.mannwhitneyu(groups["within"], groups["across"],
                                         alternative="two-sided")
        out["mannwhitney_p"] = float(pval)
        # Magnitudes matter as much as signed values, since the scopes may
        # differ in spread rather than in centre.
        stat_abs, p_abs = _stats.mannwhitneyu(np.abs(groups["within"]),
                                              np.abs(groups["across"]),
                                              alternative="two-sided")
        out["mannwhitney_p_abs"] = float(p_abs)
    else:
        out["mannwhitney_p"] = np.nan
        out["mannwhitney_p_abs"] = np.nan
    return out


def aggregate_motif_pairs(
    inters: pd.DataFrame,
    scope: str | None = None,
    value_col: str = "coef",
    motif_a_col: str = "motif_a",
    motif_b_col: str = "motif_b",
    scope_col: str = "pair_scope",
    qval_col: str = "qval",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Aggregate interaction coefficients by transcription factor pair across enhancers and genes.

    Args:
        inters: Classified interaction terms.
        scope: Restrict to "within" or "across"; both when None.
        value_col: Interaction coefficient column.
        motif_a_col: Column with the first motif.
        motif_b_col: Column with the second motif.
        scope_col: Pair scope column.
        qval_col: Adjusted p-value column.
        alpha: Significance threshold.

    Returns:
        One row per unordered factor pair with count, mean, median, range and
        number of significant occurrences.
    """
    if inters.empty:
        return pd.DataFrame(columns=["tf_a", "tf_b", "n", "mean", "median",
                                     "min", "max", "range", "n_significant"])

    work = inters.copy()
    if scope is not None:
        work = work[work[scope_col] == scope]
    work = work.dropna(subset=[value_col])
    if work.empty:
        return pd.DataFrame(columns=["tf_a", "tf_b", "n", "mean", "median",
                                     "min", "max", "range", "n_significant"])

    # Order the pair so that A-B and B-A land in the same cell.
    ordered = work.apply(
        lambda r: tuple(sorted([str(r[motif_a_col]), str(r[motif_b_col])])),
        axis=1)
    work["tf_a"] = [p[0] for p in ordered]
    work["tf_b"] = [p[1] for p in ordered]
    work["_sig"] = (work[qval_col] < alpha) if qval_col in work.columns else False

    agg = (work.groupby(["tf_a", "tf_b"])
               .agg(n=(value_col, "size"), mean=(value_col, "mean"),
                    median=(value_col, "median"), min=(value_col, "min"),
                    max=(value_col, "max"), n_significant=("_sig", "sum"))
               .reset_index())
    agg["range"] = agg["max"] - agg["min"]
    return agg.sort_values("mean").reset_index(drop=True)


def aggregate_pairs_to_enhancer(
    inters: pd.DataFrame,
    value_col: str = "coef",
    scope: str = "across",
    scope_col: str = "pair_scope",
    qval_col: str = "qval",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Aggregate motif-pair interactions to the enhancer pair they span.

    Args:
        inters: Classified interaction terms.
        value_col: Interaction coefficient column.
        scope: Pair scope to keep, normally "across".
        scope_col: Pair scope column.
        qval_col: Adjusted p-value column.
        alpha: Significance threshold.

    Returns:
        One row per enhancer pair with the sum, mean and strongest motif
        interaction, the number of motif pairs and the number significant.
    """
    if inters.empty:
        return pd.DataFrame()

    work = inters[inters[scope_col] == scope].dropna(subset=[value_col]).copy()
    if work.empty:
        return pd.DataFrame()

    work["_sig"] = (work[qval_col] < alpha) if qval_col in work.columns else False
    agg = (work.groupby(["gene", "enhA", "enhB", "enh_pair"])
               .agg(motif_sum=(value_col, "sum"),
                    motif_mean=(value_col, "mean"),
                    motif_strongest=(value_col, "min"),
                    motif_max=(value_col, "max"),
                    n_motif_pairs=(value_col, "size"),
                    n_significant=("_sig", "sum"))
               .reset_index())
    return agg


def build_motif_interdependency(
    atac_terms: pd.DataFrame,
    unit_separator: str = "|",
    include_interactions: bool = True,
) -> pd.DataFrame:
    """Effect of single motifs and motif pairs on the accessibility of every enhancer of the gene.

    Args:
        atac_terms: collect_terms output of the per-locus accessibility fits,
            main effects included.
        unit_separator: Separator between enhancer and motif in a unit label.
        include_interactions: Add the motif-pair level.

    Returns:
        One row per unit (or unit pair) and measured locus with delta, level
        ("main" or "interaction"), cis_class ("cis", "mixed" or "trans") and is_cis.
    """
    frames = []

    mains = classify_main_terms(atac_terms, unit_separator=unit_separator)
    if not mains.empty and "locus" in mains.columns:
        block = mains.rename(columns={"coef": "delta"}).copy()
        block["level"] = "main"
        block["perturbed"] = block["full_unit"]
        block["measured"] = block["locus"]
        block["unit_a"] = block["full_unit"]
        block["unit_b"] = ""
        block["motif_b"] = ""
        block["enh_b"] = ""
        block["motif_a"] = block["motif"]
        block["enh_a"] = block["enh"]
        block["cis_class"] = np.where(block["is_cis"], "cis", "trans")
        block["pair_scope"] = ""
        frames.append(block)
    else:
        warnings.warn("No per-locus main effects found; "
                      "was collect_terms run on fit_all_loci output?")

    if include_interactions:
        pairs = classify_interaction_terms(atac_terms,
                                           unit_separator=unit_separator)
        if not pairs.empty and "locus" in pairs.columns:
            block = pairs.rename(columns={"coef": "delta"}).copy()
            block["level"] = "interaction"
            block["measured"] = block["locus"]
            # A pair is named by both units, joined so that the key cannot be
            # confused with either an interaction term or a dimer motif label.
            block["perturbed"] = block["full_a"] + " x " + block["full_b"]

            in_a = block["gene"] + "_" + block["enh_a"] == block["measured"]
            in_b = block["gene"] + "_" + block["enh_b"] == block["measured"]
            block["cis_class"] = np.select(
                [in_a & in_b, in_a ^ in_b],
                ["cis", "mixed"],
                default="trans",
            )
            block["is_cis"] = block["cis_class"] == "cis"
            frames.append(block)

    if not frames:
        return pd.DataFrame()

    keep = ["gene", "level", "perturbed", "measured", "unit_a", "unit_b",
            "enh_a", "enh_b", "motif_a", "motif_b", "cis_class", "is_cis",
            "pair_scope", "delta", "se", "pval", "qval"]
    out = pd.concat(frames, ignore_index=True)
    return out[[c for c in keep if c in out.columns]].reset_index(drop=True)


def attach_motif_distance(
    inters: pd.DataFrame,
    seqlets: pd.DataFrame,
    label_col: str = "label",
    start_col: str = "genomic_start",
    end_col: str = "genomic_end",
) -> pd.DataFrame:
    """Add the genomic distance between the midpoints of the two motifs of each pair.

    Args:
        inters: Classified interaction terms with full_a and full_b.
        seqlets: Selected seqlets with gene-qualified labels and genomic coordinates.
        label_col: Unit label column in seqlets.
        start_col: Genomic start column.
        end_col: Genomic end column.

    Returns:
        Copy of inters with distance_bp, missing where a motif is not matched.
    """
    if inters.empty or seqlets.empty:
        return inters.copy()
    if not {start_col, end_col, label_col}.issubset(seqlets.columns):
        warnings.warn("Seqlet table lacks genomic coordinates; "
                      "distance not attached.")
        return inters.copy()

    midpoint = (seqlets.drop_duplicates(label_col)
                       .set_index(label_col)
                       .eval(f"({start_col} + {end_col}) / 2"))
    out = inters.copy()
    out["mid_a"] = out["full_a"].map(midpoint)
    out["mid_b"] = out["full_b"].map(midpoint)
    out["distance_bp"] = (out["mid_a"] - out["mid_b"]).abs()

    missing = int(out["distance_bp"].isna().sum())
    if missing:
        warnings.warn(f"{missing} pair(s) could not be matched to seqlet "
                      f"coordinates; check that labels agree.")
    return out