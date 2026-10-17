"""Per-gene log-link GLM for enhancer knockouts, with coefficient forest plots and heatmaps.

Original enhancer-level implementation used by notebook 01; glm.py generalises
the same model to other perturbation units.
"""

import ast, itertools, warnings
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial as SM_NB
from scipy import stats

def build_design_matrix(
    scr_lists: list[list[str]],
    enhancers: list[str],
    max_order: int,
) -> tuple[np.ndarray, list[str]]:
    """Build an intercept, main-effect and interaction design matrix.

    A term for a set of enhancers is 1 when every enhancer in the set is scrambled.

    Args:
        scr_lists: Scrambled enhancers per observation.
        enhancers: Ordered enhancers of the gene.
        max_order: Highest interaction order (1 main effects, 2 pairs, 3 triplets).

    Returns:
        Tuple of (design matrix with intercept column, term names).
    """
    n_enh     = len(enhancers)
    max_order = min(max_order, n_enh)

    # short labels for readability
    def short(e: str) -> str: return e.split("_", 1)[-1] if "_" in e else e

    term_groups = []   # list of (name, frozenset_of_enhancers)
    for order in range(1, max_order + 1):
        for combo in itertools.combinations(range(n_enh), order):
            enhs  = [enhancers[i] for i in combo]
            label = ":".join(short(e) for e in enhs)
            term_groups.append((label, frozenset(enhs)))

    scr_sets = [frozenset(sl) for sl in scr_lists]
    n_obs    = len(scr_sets)
    n_terms  = len(term_groups)

    X = np.zeros((n_obs, n_terms))
    for j, (_, term_set) in enumerate(term_groups):
        for i, scr_set in enumerate(scr_sets):
            X[i, j] = 1.0 if term_set <= scr_set else 0.0

    X          = np.column_stack([np.ones(n_obs), X])
    term_names = ["intercept"] + [t[0] for t in term_groups]

    return X, term_names

def _fit_glm(
    y: np.ndarray,
    X: np.ndarray,
    term_names: list[str],
    family: str,
    round_nb: bool,
) -> pd.DataFrame | None:
    """Fit one negative binomial or Gamma log-link GLM and return a tidy coefficient table."""
    try:
        if family == "nb":
            y_fit = np.round(y).astype(int) if round_nb else y.astype(int)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = SM_NB(y_fit, X).fit(
                    method="bfgs", disp=False, maxiter=200
                )

        elif family == "gamma":
            y_fit = np.clip(y, 1e-6, None)   # gamma requires strictly positive
            result = sm.GLM(
                y_fit, X,
                family=sm.families.Gamma(link=sm.families.links.Log()),
            ).fit(disp=False)

        else:
            raise ValueError(f"Unknown family '{family}'. Use 'nb' or 'gamma'.")

    except Exception as e:
        warnings.warn(f"GLM fit failed: {e}")
        return None

    # statsmodels may return pandas objects; use plain arrays
    ci     = np.asarray(result.conf_int())
    pvals  = np.asarray(result.pvalues)
    params = np.asarray(result.params)
    bse    = np.asarray(result.bse)

    # The NB fit adds a dispersion parameter ("alpha") after the design terms.
    local_terms = list(term_names)
    if len(params) == len(local_terms) + 1:
        local_terms.append("alpha")
    elif len(params) != len(local_terms):
        # Any other length mismatch means the table cannot be aligned
        warnings.warn(f"Parameter length mismatch: expected {len(local_terms)}, got {len(params)}")
        return None

    return pd.DataFrame(dict(
        term   = local_terms,
        coef   = params,
        se     = bse,
        ci_lo  = ci[:, 0], 
        ci_hi  = ci[:, 1],
        pval   = pvals,
        family = family,
    ))

import ast
import numpy as np

def parse_sequence_column(v: Any) -> list:
    """Convert a list-like cell (list, array or its string form) into a Python list.

    Args:
        v: Cell value.

    Returns:
        The value as a list; an empty list when it cannot be parsed.
    """
    if isinstance(v, list):
        return v
        
    if isinstance(v, np.ndarray):
        return v.tolist()
        
    if isinstance(v, str):
        # repr of a numpy array cannot be parsed by literal_eval; treat as empty
        if v.startswith("array("):
            return []
            
        try:
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return []
            
    # NaN and other non-list values
    return []

def fit_interaction_glm(
    df_raw: pd.DataFrame,
    gene: str,
    family: str = "nb",
    max_order: int = 2,
    rna_col: str = "RNA_counts",
    round_nb: bool = True,
    verbose: bool = True,
) -> dict:
    """Fit the interaction GLM of one gene on the unaggregated scramble table.

    Args:
        df_raw: Scramble results, one row per shuffle replicate.
        gene: Gene in the "name" column.
        family: "nb" or "gamma".
        max_order: Highest interaction order to include.
        rna_col: Response column.
        round_nb: Round the response to integers for the negative binomial.
        verbose: Print the coefficient table.

    Returns:
        Dictionary with gene, family, max_order, coef_table, enhancers, n_obs and
        converged.
    """
    grp = df_raw[df_raw["name"] == gene].copy().reset_index(drop=True)
    if grp.empty:
        raise ValueError(f"Gene '{gene}' not found.")
        
    grp["scr_list"] = grp["scrambled_seqlets"].apply(parse_sequence_column)
    grp["enh_list"] = grp["target_enhancer"].apply(parse_sequence_column)

    # enhancer order from the baseline row
    base = grp[grp["n_scrambled"].astype(int) == 0].iloc[0]
    enhancers = base["enh_list"]

    # response
    y = grp[rna_col].astype(float).values

    # drop NaN responses
    valid = np.isfinite(y)
    if valid.sum() < len(enhancers) + 2:
        warnings.warn(f"{gene}: too few valid observations ({valid.sum()}).")
        return dict(gene=gene, converged=False, coef_table=None,
                    enhancers=enhancers, n_obs=int(valid.sum()),
                    family=family, max_order=max_order)

    y   = y[valid]
    scr = grp.loc[valid, "scr_list"].tolist()

    X, term_names = build_design_matrix(scr, enhancers, max_order)

    coef_table = _fit_glm(y, X, term_names, family, round_nb)
    converged  = coef_table is not None

    if verbose and converged:
        interaction_terms = coef_table[~coef_table["term"].isin(
            ["intercept"] + [t.split("_", 1)[-1] if "_" in t else t
             for t in enhancers]
        )]
        print(f"\n{'─'*60}")
        print(f"  {gene}  |  family={family}  |  max_order={max_order}"
              f"  |  n={len(y)}")
        print(f"{'─'*60}")
        print(coef_table.to_string(index=False, float_format="{:.4f}".format))

    return dict(
        gene       = gene,
        family     = family,
        max_order  = max_order,
        coef_table = coef_table,
        enhancers  = enhancers,
        n_obs      = len(y),
        converged  = converged,
    )

def fit_all_genes(
    df_raw: pd.DataFrame,
    genes: list[str] | None = None,
    family: str = "nb",
    max_order: int = 2,
    rna_col: str = "RNA_counts",
    round_nb: bool = True,
    verbose: bool = False,
) -> dict[str, dict]:
    """Fit fit_interaction_glm for every gene.

    Args:
        df_raw: Scramble results, one row per shuffle replicate.
        genes: Genes to fit; all genes in "name" when None.
        family: "nb" or "gamma".
        max_order: Highest interaction order to include.
        rna_col: Response column.
        round_nb: Round the response to integers for the negative binomial.
        verbose: Print each coefficient table.

    Returns:
        Mapping of gene to its fit_interaction_glm result.
    """
    if genes is None:
        genes = list(df_raw["name"].unique())

    results = {}
    for gene in genes:
        try:
            results[gene] = fit_interaction_glm(
                df_raw, gene,
                family=family, max_order=max_order,
                rna_col=rna_col, round_nb=round_nb, verbose=verbose,
            )
        except Exception as e:
            warnings.warn(f"Failed for {gene}: {e}")
    return results


def collect_interaction_terms(results: dict[str, dict]) -> pd.DataFrame:
    """Stack the interaction coefficients of all genes into one table.

    Args:
        results: Output of fit_all_genes.

    Returns:
        One row per interaction term with its gene and order.
    """
    rows = []
    for gene, res in results.items():
        if not res["converged"] or res["coef_table"] is None:
            continue
        # main effects have no ":" in the term name
        inter = res["coef_table"][res["coef_table"]["term"].str.contains(":")]
        for _, r in inter.iterrows():
            rows.append({**r.to_dict(), "gene": gene,
                         "order": r["term"].count(":") + 1})
    return pd.DataFrame(rows)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def _prepare_crispr_table(
    gene: str, 
    single_df: pd.DataFrame | None, 
    inter_df: pd.DataFrame | None
) -> pd.DataFrame | None:
    """Reshape measured CRISPRi effects of one gene into the coefficient-table layout."""
    rows = []
    prefix = f"{gene}_"

    # Single-enhancer effects become main-effect terms
    if single_df is not None and not single_df.empty:
        gene_single = single_df[single_df["enhancer"].str.startswith(prefix, na=False)]
        for _, row in gene_single.iterrows():
            term = row["enhancer"].replace(prefix, "")
            rows.append({"term": term, "coef": row["interaction_beta"]})

    # Pair effects become "enhA:enhB" interaction terms
    if inter_df is not None and not inter_df.empty:
        gene_inter = inter_df[
            inter_df["enhA"].str.startswith(prefix, na=False) &
            inter_df["enhB"].str.startswith(prefix, na=False)
        ]
        for _, row in gene_inter.iterrows():
            termA = row["enhA"].replace(prefix, "")
            termB = row["enhB"].replace(prefix, "")
            term = f"{termA}:{termB}"
            rows.append({"term": term, "coef": row["interaction_beta"]})

    if not rows:
        return None

    tbl = pd.DataFrame(rows)
    # NaN CI and p-value: the forest panel then draws no error bars or p-text
    tbl["ci_lo"] = np.nan
    tbl["ci_hi"] = np.nan
    tbl["pval"]  = np.nan
    
    return tbl


def _plot_single_forest(
    ax: plt.Axes,
    coef_table: pd.DataFrame,
    title: str,
    skip_intercept: bool = True,
    alpha_thresh: float = 0.05,
) -> plt.Axes:
    """Draw one coefficient forest on an axis."""
    tbl = coef_table.copy()
    if skip_intercept and "intercept" in tbl["term"].values:
        tbl = tbl[tbl["term"] != "intercept"]
    tbl = tbl.reset_index(drop=True)

    y = np.arange(len(tbl))
    ax.set_facecolor("#F7F7F5")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CCCCCC")

    def _color(row: pd.Series) -> str:
        is_inter = ":" in row["term"]
        # No p-value means measured CRISPRi data: colour by term type only
        if pd.isna(row.get("pval")):
            return "#E84855" if is_inter else "#2176AE"
            
        sig = row["pval"] < alpha_thresh
        if not sig:  return "#BBBBBB"
        if is_inter: return "#E84855"
        return "#2176AE"

    colors = [_color(r) for _, r in tbl.iterrows()]
    ax.axvline(0, color="gray", lw=1, ls="--", zorder=0)

    for i, (_, row) in enumerate(tbl.iterrows()):
        has_ci = "ci_lo" in row and "ci_hi" in row and pd.notna(row["ci_lo"])
        if has_ci:
            ax.plot([row["ci_lo"], row["ci_hi"]], [i, i],
                    color=colors[i], lw=1.8, solid_capstyle="round", zorder=2)
            
        ax.scatter(row["coef"], i, color=colors[i],
                   s=55, zorder=3, edgecolors="white", lw=0.5)
                   
        pval = row.get("pval")
        if pd.notna(pval):
            pstr = (f"p={pval:.2e}" if pval < 0.001 else f"p={pval:.3f}")
            ax.text(row["ci_hi"] + 0.01, i, pstr,
                    va="center", fontsize=6.5, color="#555")

    ax.set_yticks(y)
    ax.set_yticklabels(tbl["term"], fontsize=8.5)
    ax.set_xlabel("Coefficient (log scale)", fontsize=9)
    ax.set_title(title, fontweight="bold", fontsize=9.5)
    return ax


def plot_coef_forest(
    result: dict,
    ax: plt.Axes | None = None,
    skip_intercept: bool = True,
    alpha_thresh: float = 0.05,
) -> plt.Axes:
    """Forest plot of the GLM coefficients of one fit.

    Args:
        result: Output of fit_interaction_glm or one locus of fit_interaction_glm_atac.
        ax: Axis to draw on; a new figure is created when None.
        skip_intercept: Hide the intercept.
        alpha_thresh: Significance threshold for colouring.

    Returns:
        The axis.
    """
    if not result.get("converged", True):
        if ax is None: _, ax = plt.subplots()
        ax.text(0.5, 0.5, "Model did not converge",
                ha="center", va="center", transform=ax.transAxes, color="red")
        return ax

    if ax is None:
        n_terms = len(result["coef_table"]) - (1 if skip_intercept else 0)
        _, ax = plt.subplots(figsize=(6, max(3, n_terms * 0.38)))

    title = (f"{result['gene']}  |  {result['family'].upper()}  "
             f"|  max_order={result['max_order']}  |  n={result['n_obs']}")
    _plot_single_forest(ax, result["coef_table"], title, skip_intercept, alpha_thresh)

    ax.legend(handles=[
        mpatches.Patch(color="#2176AE", label=f"main effect (p<{alpha_thresh})"),
        mpatches.Patch(color="#E84855", label=f"interaction  (p<{alpha_thresh})"),
        mpatches.Patch(color="#BBBBBB", label="ns"),
    ], fontsize=7.5, loc="upper left", framealpha=0.85)
    return ax


def plot_coef_forest_grid(
    rna_result: dict | None = None,
    atac_result: dict | None = None,
    crispr_single: pd.DataFrame | None = None,
    crispr_inter: pd.DataFrame | None = None,
    skip_intercept: bool = True,
    alpha_thresh: float = 0.05,
    panel_width: float = 3.2,
) -> plt.Figure:
    """Side-by-side forest plots of the RNA and ATAC fits and the measured CRISPRi effects.

    Args:
        rna_result: RNA fit of one gene.
        atac_result: ATAC fit of the same gene.
        crispr_single: Measured single-enhancer effects.
        crispr_inter: Measured pairwise interactions.
        skip_intercept: Hide the intercept.
        alpha_thresh: Significance threshold for colouring.
        panel_width: Figure width per panel in inches.

    Returns:
        The figure.
    """
    if rna_result is None and atac_result is None:
        raise ValueError("Provide at least one of rna_result / atac_result.")

    gene = (rna_result or atac_result)["gene"]
    panels = []   

    if rna_result is not None:
        panels.append(("RNA" if rna_result.get("converged") else "RNA (no convergence)",
                       rna_result["coef_table"] if rna_result.get("converged") else None))

    if atac_result is not None:
        for locus in atac_result.get("enhancers", []):
            short = locus.split("_", 1)[-1] if "_" in locus else locus
            panels.append((f"ATAC @ {short}", atac_result["per_locus"].get(locus)))

    # Add a CRISPRi panel when measured data are supplied
    if crispr_single is not None or crispr_inter is not None:
        crispr_tbl = _prepare_crispr_table(gene, crispr_single, crispr_inter)
        if crispr_tbl is not None:
            panels.append(("CRISPR Validation", crispr_tbl))

    n_panels = len(panels)
    
    # Figure height scales with the longest coefficient table
    max_terms = max(
        (len(t[t["term"] != "intercept"]) if (skip_intercept and t is not None and "intercept" in t["term"].values)
         else (len(t) if t is not None else 1))
        for _, t in panels
    )
    panel_height = max(3, max_terms * 0.38)

    fig, axes = plt.subplots(1, n_panels, figsize=(panel_width * n_panels, panel_height))
    axes = np.atleast_1d(axes)

    for ax, (title, tbl) in zip(axes, panels):
        if tbl is None:
            ax.text(0.5, 0.5, "Model did not\nconverge\n(or missing data)", 
                    ha="center", va="center", transform=ax.transAxes, 
                    color="red", fontsize=9)
            ax.set_facecolor("#F7F7F5")
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(title, fontweight="bold", fontsize=9.5)
            continue
            
        _plot_single_forest(ax, tbl, title, skip_intercept, alpha_thresh)

    fig.suptitle(f"{gene} | coefficient forests", fontweight="bold", fontsize=12, y=1.02)
    fig.legend(handles=[
        mpatches.Patch(color="#2176AE", label=f"main effect (GLM: p<{alpha_thresh})"),
        mpatches.Patch(color="#E84855", label=f"interaction  (GLM: p<{alpha_thresh})"),
        mpatches.Patch(color="#BBBBBB", label="GLM: ns"),
    ], fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, framealpha=0.85)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def plot_interaction_heatmap(
    results: dict[str, dict],
    ax: plt.Axes | None = None,
    alpha_thresh: float = 0.05,
    value: str = "coef",
) -> plt.Axes:
    """Heatmap of interaction terms, genes as rows and terms as columns.

    Args:
        results: Output of fit_all_genes.
        ax: Axis to draw on; a new figure is created when None.
        alpha_thresh: Significance threshold; non-significant cells are hatched.
        value: "coef" for coefficients or "pval" for -log10 p-values.

    Returns:
        The axis.
    """
    inter_df = collect_interaction_terms(results)
    if inter_df.empty:
        print("No interaction terms found.")
        return ax

    pivot = inter_df.pivot_table(
        index="gene", columns="term", values=value, aggfunc="first"
    )
    piv_p = inter_df.pivot_table(
        index="gene", columns="term", values="pval", aggfunc="first"
    )

    if value == "pval":
        plot_mat = -np.log10(pivot.values.astype(float))
        cbar_label = "−log₁₀(p)"
        cmap, vmin, vmax = "Reds", 0, None
    else:
        plot_mat = pivot.values.astype(float)
        cbar_label = "β (interaction coefficient)"
        v = np.nanmax(np.abs(plot_mat))
        cmap, vmin, vmax = "RdBu_r", -v, v

    if ax is None:
        h = max(3, len(pivot) * 0.5)
        w = max(4, len(pivot.columns) * 0.9)
        _, ax = plt.subplots(figsize=(w, h))

    ax.set_facecolor("#F7F7F5")
    cmap_obj = plt.cm.get_cmap(cmap).copy()
    cmap_obj.set_bad("#E0E0E0")

    im = ax.imshow(
        np.ma.masked_invalid(plot_mat),
        cmap=cmap_obj, vmin=vmin, vmax=vmax, aspect="auto",
    )

    # hatch non-significant cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            p = piv_p.values[i, j]
            if np.isfinite(p) and p >= alpha_thresh:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, hatch="///", edgecolor="#AAAAAA",
                    linewidth=0, zorder=3
                ))
            v_ = plot_mat[i, j]
            if np.isfinite(v_):
                brightness = abs(v_) / (abs(vmax) + 1e-9) if vmax else 0
                tc = "white" if brightness > 0.65 else "#222"
                ax.text(j, i, f"{v_:.2f}", ha="center", va="center",
                        fontsize=7, color=tc)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=40, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8.5)

    cb = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label(cbar_label, fontsize=8.5)

    ax.set_title(
        f"Interaction coefficients  |  "
        f"hatched = p ≥ {alpha_thresh}  |  "
        f"family = {next(iter(results.values()))['family'].upper()}",
        fontweight="bold", fontsize=10,
    )
    return ax

def fit_interaction_glm_atac(
    df_raw: pd.DataFrame,
    gene: str,
    family: str = "gamma",
    max_order: int = 2,
    atac_col: str = "ATAC_counts",
    round_nb: bool = True,
    verbose: bool = True,
) -> dict:
    """Fit the interaction GLM separately for the accessibility of each enhancer of a gene.

    The design matrix is shared across loci; only the response changes.

    Args:
        df_raw: Scramble results with a per-enhancer ATAC list column.
        gene: Gene in the "name" column.
        family: "nb" or "gamma".
        max_order: Highest interaction order to include.
        atac_col: Column with the per-enhancer accessibility list.
        round_nb: Round the response to integers for the negative binomial.
        verbose: Print each coefficient table.

    Returns:
        Dictionary with gene, family, max_order, enhancers, n_obs, per_locus
        (coefficient table per locus) and coef_table (all loci stacked).
    """
    grp = df_raw[df_raw["name"] == gene].copy().reset_index(drop=True)
    if grp.empty:
        raise ValueError(f"Gene '{gene}' not found.")

    grp["scr_list"] = grp["scrambled_seqlets"].apply(parse_sequence_column)
    grp["enh_list"] = grp["target_enhancer"].apply(parse_sequence_column)
    grp["atac_arr"] = grp[atac_col].apply(parse_sequence_column)

    base      = grp[grp["n_scrambled"].astype(int) == 0].iloc[0]
    enhancers = base["enh_list"]            # fixed, full set of measured loci
    n_enh     = len(enhancers)

    # sanity check: every row should report ATAC for the same n_enh loci,
    # in the same order, since target_enhancer is constant for this gene
    bad_rows = grp["atac_arr"].apply(len) != n_enh
    if bad_rows.any():
        warnings.warn(
            f"{gene}: {bad_rows.sum()} row(s) with ATAC vector length "
            f"≠ {n_enh}; dropping them."
        )
        grp = grp[~bad_rows].reset_index(drop=True)

    scr = grp["scr_list"].tolist()
    X, term_names = build_design_matrix(scr, enhancers, max_order)

    per_locus  = {}
    all_tables = []

    for k, locus in enumerate(enhancers):
        y = np.array([row[k] for row in grp["atac_arr"]], dtype=float)
        valid = np.isfinite(y)

        if valid.sum() < X.shape[1] + 1:
            warnings.warn(
                f"{gene} / {locus}: too few valid observations "
                f"({valid.sum()}) for {X.shape[1]} terms."
            )
            per_locus[locus] = None
            continue

        coef_table = _fit_glm(y[valid], X[valid], term_names, family, round_nb)
        per_locus[locus] = coef_table

        if coef_table is not None:
            coef_table = coef_table.copy()
            coef_table.insert(0, "measured_enhancer", locus)
            all_tables.append(coef_table)

        if verbose and coef_table is not None:
            print(f"\n{'─'*60}")
            print(f"  {gene}  |  locus={locus}  |  family={family}"
                  f"  |  max_order={max_order}  |  n={int(valid.sum())}")
            print(f"{'─'*60}")
            print(coef_table.drop(columns="measured_enhancer")
                  .to_string(index=False, float_format="{:.4f}".format))

    combined = pd.concat(all_tables, ignore_index=True) if all_tables else None

    return dict(
        gene       = gene,
        family     = family,
        max_order  = max_order,
        enhancers  = enhancers,
        n_obs      = len(grp),
        per_locus  = per_locus,
        coef_table = combined,
    )

def fit_interaction_glm_atac_all_genes(
    df_raw: pd.DataFrame,
    family: str = "gamma",
    max_order: int = 2,
    atac_col: str = "ATAC_counts",
    round_nb: bool = True,
    verbose: bool = False,
    genes: list[str] | None = None,
) -> dict[str, dict]:
    """Fit fit_interaction_glm_atac for every gene.

    Args:
        df_raw: Scramble results with a per-enhancer ATAC list column.
        family: "nb" or "gamma".
        max_order: Highest interaction order to include.
        atac_col: Column with the per-enhancer accessibility list.
        round_nb: Round the response to integers for the negative binomial.
        verbose: Print each coefficient table.
        genes: Genes to fit; all genes in "name" when None.

    Returns:
        Mapping of gene to its result; genes that fail get a converged=False stub.
    """
    gene_list = genes if genes is not None else df_raw["name"].unique()
    results = {}
    for gene in gene_list:
        try:
            results[gene] = fit_interaction_glm_atac(
                df_raw, gene=gene, family=family, max_order=max_order,
                atac_col=atac_col, round_nb=round_nb, verbose=verbose,
            )
        except Exception as e:
            warnings.warn(f"{gene}: ATAC fit failed entirely — {e}")
            results[gene] = dict(
                gene=gene, converged=False, per_locus=None,
                coef_table=None, enhancers=None, n_obs=0,
                family=family, max_order=max_order,
            )
    return results


