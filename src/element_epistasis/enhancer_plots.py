"""Figures for the enhancer-level in silico knockout.

Each panel function takes an enhancer_analysis or GLM table and an optional
axis and returns the axis. b_i is the main effect, b_ij the interaction
(negative: super-additive); significance markers use BH-adjusted p-values.
"""

from __future__ import annotations

import ast
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats

from element_epistasis.plot_style import (
    CMAP_INTERACTION, LBL_DIRECTION, PAL, add_direction_annotations,
    empty_panel, panel_letters, pformat, stars, style_ax,
)
from element_epistasis.plot_style import label_points as place_labels

LBL_MAIN = "GLM main effect  $\\beta_i$   (log scale)"
LBL_INTER = "GLM interaction  $\\beta_{ij}$   (log scale)"

# Cells with no estimate. Pale blue rather than grey, because grey sits close
# to the midpoint of the diverging colormap and an empty cell would otherwise
# read as a coefficient near zero.
EMPTY_CELL = "#D7E6F2"


# --------------------------------------------------------------------------- #
# Model choice
# --------------------------------------------------------------------------- #

_METRIC_LABELS = {
    "mean_se": "Mean coefficient standard error",
    "aic": "AIC",
    "deviance": "Deviance",
    "n_terms": "Number of fitted terms",
}


def plot_model_comparison(
    comparison: pd.DataFrame,
    ax: plt.Axes | None = None,
    metric: str = "mean_se",
) -> plt.Axes:
    """Compare a fit diagnostic across model families and interaction orders.

    Args:
        comparison: build_model_comparison output.
        ax: Axis to draw on; created when None.
        metric: Diagnostic column to plot.

    Returns:
        The axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    if comparison.empty:
        return empty_panel(ax, "No model comparison table", "Model choice")

    note = ""
    if metric not in comparison.columns or comparison[metric].isna().all():
        if "mean_se" in comparison.columns and comparison["mean_se"].notna().any():
            note = f"{metric} unavailable; showing mean_se"
            metric = "mean_se"
        else:
            return empty_panel(ax, f"No finite '{metric}' values", "Model choice")

    df = comparison.dropna(subset=[metric])
    if df.empty:
        return empty_panel(ax, f"No finite '{metric}' values", "Model choice")

    style_ax(ax)
    settings = list(dict.fromkeys(df["setting"]))
    data = [df.loc[df["setting"] == s, metric].to_numpy() for s in settings]
    positions = np.arange(len(settings))

    populated = [i for i, d in enumerate(data) if len(d)]
    if populated:
        parts = ax.violinplot([data[i] for i in populated],
                              positions=positions[populated],
                              showmedians=True, showextrema=False, widths=0.6)
        for body in parts["bodies"]:
            body.set_facecolor(PAL["atac"])
            body.set_alpha(0.4)
            body.set_edgecolor("white")
        parts["cmedians"].set_colors("white")
        parts["cmedians"].set_linewidth(2)

    rng = np.random.default_rng(3)
    for i, values in enumerate(data):
        if not len(values):
            continue
        ax.scatter(positions[i] + rng.uniform(-0.07, 0.07, len(values)), values,
                   color=PAL["atac"], s=24, alpha=0.8,
                   edgecolors="white", lw=0.4, zorder=3)

    # Convergence failures are the decisive argument against a family and are
    # counted from the unfiltered table, since a failed fit has no metric.
    top = ax.get_ylim()[1]
    for i, setting in enumerate(settings):
        failed = int((~comparison.loc[comparison["setting"] == setting,
                                      "converged"]).sum())
        if failed:
            ax.text(positions[i], top, f"{failed} failed", ha="center",
                    va="bottom", fontsize=7, color=PAL["rna"])

    if note:
        ax.text(0.03, 0.96, note, transform=ax.transAxes, va="top",
                fontsize=7, color="#777", style="italic")

    ax.set_xticks(positions)
    ax.set_xticklabels(settings, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel(_METRIC_LABELS.get(metric, metric), fontsize=9)
    ax.set_title("Model family and interaction order",
                 fontweight="bold", fontsize=10)
    return ax


# --------------------------------------------------------------------------- #
# Per-gene coefficient forests
# --------------------------------------------------------------------------- #

def _forest_panel(
    ax: plt.Axes,
    coef_table: pd.DataFrame,
    title: str,
    skip_intercept: bool = True,
    qval_col: str = "qval",
    alpha: float = 0.05,
) -> plt.Axes:
    """Draw one coefficient forest, interactions coloured by sign."""
    tbl = coef_table.copy()
    if skip_intercept and "intercept" in tbl["term"].values:
        tbl = tbl[tbl["term"] != "intercept"]
    tbl = tbl.reset_index(drop=True)

    style_ax(ax)
    ax.axvline(0, color="gray", lw=1, ls="--", zorder=0)

    for i, (_, row) in enumerate(tbl.iterrows()):
        is_inter = ":" in str(row["term"])
        q = row.get(qval_col, row.get("pval", np.nan))
        significant = pd.isna(q) or q < alpha

        if not significant:
            color = PAL["neut"]
        elif is_inter:
            # Direction carries the meaning for interaction terms.
            color = PAL["supra"] if row["coef"] < 0 else PAL["sub"]
        else:
            color = PAL["atac"]

        has_ci = pd.notna(row.get("ci_lo")) and pd.notna(row.get("ci_hi"))
        if has_ci:
            ax.plot([row["ci_lo"], row["ci_hi"]], [i, i], color=color,
                    lw=1.8, solid_capstyle="round", zorder=2)
        ax.scatter(row["coef"], i, color=color, s=55, zorder=3,
                   edgecolors="white", lw=0.5)

        marker = stars(q) if pd.notna(q) else ""
        if marker:
            anchor = row["ci_hi"] if has_ci else row["coef"]
            ax.text(anchor + 0.01, i, marker, va="center", fontsize=8,
                    color=PAL["ink"])

    ax.set_yticks(np.arange(len(tbl)))
    ax.set_yticklabels(tbl["term"], fontsize=8.5)
    ax.set_xlabel(f"$\\beta$   (log scale)\n{LBL_DIRECTION}", fontsize=8.5)
    ax.set_title(title, fontweight="bold", fontsize=9.5)
    return ax


def plot_coef_forest(
    result: dict,
    ax: plt.Axes | None = None,
    skip_intercept: bool = True,
    qval_col: str = "qval",
    alpha: float = 0.05,
) -> plt.Axes:
    """Forest plot of all GLM coefficients of one gene.

    Args:
        result: One entry of fit_all_genes.
        ax: Axis to draw on; created when None.
        skip_intercept: Hide the intercept.
        qval_col: Adjusted p-value column.
        alpha: Significance threshold.

    Returns:
        The axis.
    """
    if not result.get("converged", True):
        if ax is None:
            _, ax = plt.subplots()
        return empty_panel(ax, "Model did not converge",
                           result.get("gene", "unknown"))

    table = result["coef_table"]
    if ax is None:
        n_terms = len(table) - (1 if skip_intercept else 0)
        _, ax = plt.subplots(figsize=(6, max(3, n_terms * 0.38)))

    title = (f"{result['gene']}  |  {result.get('family', '?').upper()}  "
             f"|  order {result.get('max_order', '?')}  "
             f"|  n = {result.get('n_obs', '?')}")
    _forest_panel(ax, table, title, skip_intercept, qval_col, alpha)

    ax.legend(handles=[
        mpatches.Patch(color=PAL["atac"], label="main effect"),
        mpatches.Patch(color=PAL["supra"], label="interaction, super-additive"),
        mpatches.Patch(color=PAL["sub"], label="interaction, sub-additive"),
        mpatches.Patch(color=PAL["neut"], label=f"FDR $\\geq$ {alpha:g}"),
    ], fontsize=7, loc="best", framealpha=0.85)
    return ax


def plot_coef_forest_grid(
    rna_result: dict | None = None,
    atac_result: dict | None = None,
    crispri_table: pd.DataFrame | None = None,
    skip_intercept: bool = True,
    qval_col: str = "qval",
    alpha: float = 0.05,
    panel_width: float = 3.2,
) -> plt.Figure:
    """Side-by-side forests of the RNA fit, the per-locus ATAC fits and the measured CRISPRi effects.

    Args:
        rna_result: RNA GLM result of one gene.
        atac_result: ATAC GLM result with per_locus.
        crispri_table: Measured coefficients in coefficient-table layout.
        skip_intercept: Hide the intercepts.
        qval_col: Adjusted p-value column.
        alpha: Significance threshold.
        panel_width: Width per panel in inches.

    Returns:
        The figure.
    """
    if rna_result is None and atac_result is None:
        raise ValueError("Provide at least one of rna_result or atac_result.")

    gene = (rna_result or atac_result)["gene"]
    panels = []

    if rna_result is not None and rna_result.get("converged", True):
        panels.append((f"RNA @ {gene}", rna_result["coef_table"]))

    if atac_result is not None:
        for locus in atac_result.get("enhancers", []):
            short = locus.split("_", 1)[-1] if "_" in locus else locus
            table = atac_result.get("per_locus", {}).get(locus)
            if table is not None:
                panels.append((f"ATAC @ {short}", table))

    if crispri_table is not None and not crispri_table.empty:
        panels.append(("Measured CRISPRi", crispri_table))

    if not panels:
        raise ValueError("Nothing to plot.")

    height = max(3.0, max(len(t) for _, t in panels) * 0.38)
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(panel_width * len(panels), height))
    axes = np.atleast_1d(axes)

    for ax, (title, table) in zip(axes, panels):
        _forest_panel(ax, table, title, skip_intercept, qval_col, alpha)

    fig.suptitle(f"Coefficient comparison for {gene}",
                 fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


# --------------------------------------------------------------------------- #
# Interaction structure
# --------------------------------------------------------------------------- #

def plot_interaction_volcano(
    inters: pd.DataFrame,
    ax: plt.Axes | None = None,
    qval_col: str = "qval",
    alpha: float = 0.05,
    label_top: int = 10,
    label_col: str | None = None,
) -> plt.Axes:
    """Volcano plot of enhancer-pair interactions against adjusted p-value.

    Args:
        inters: Interaction terms after BH correction.
        ax: Axis to draw on; created when None.
        qval_col: Adjusted p-value column.
        alpha: Significance threshold line.
        label_top: Number of pairs labelled per direction.
        label_col: Label column; defaults to gene plus term.

    Returns:
        The axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5))

    if inters.empty or qval_col not in inters.columns:
        return empty_panel(ax, "No corrected interaction terms",
                           "Enhancer pair interactions")

    df = inters.dropna(subset=["coef", qval_col]).copy()
    if df.empty:
        return empty_panel(ax, "No corrected interaction terms",
                           "Enhancer pair interactions")

    style_ax(ax)
    df["neglog_q"] = -np.log10(df[qval_col].clip(lower=1e-300))
    hit = df[qval_col] < alpha
    colors = np.where(~hit, PAL["neut"],
                      np.where(df["coef"] < 0, PAL["supra"], PAL["sub"]))

    ax.scatter(df["coef"], df["neglog_q"], c=colors, s=45, alpha=0.85,
               edgecolors="white", lw=0.5, zorder=3)
    ax.axhline(-np.log10(alpha), color="gray", lw=1, ls="--", zorder=0)
    ax.axvline(0, color="gray", lw=0.8, ls=":", zorder=0)

    # A term alone is ambiguous across genes, so qualify it unless told otherwise.
    if label_col and label_col in df.columns:
        df["_label"] = df[label_col].astype(str)
    elif {"gene", "term"}.issubset(df.columns):
        df["_label"] = df["gene"].astype(str) + "\n" + df["term"].astype(str)
    elif {"full_a", "full_b"}.issubset(df.columns):
        df["_label"] = df["full_a"].astype(str) + "\n" + df["full_b"].astype(str)
    else:
        df["_label"] = df.index.astype(str)

    # Label each direction separately, so the super-additive and sub-additive
    # extremes are both shown rather than whichever side has larger effects.
    if label_top and hit.any():
        significant = df[hit]
        chosen = pd.concat([
            significant[significant["coef"] < 0].nsmallest(label_top, "coef"),
            significant[significant["coef"] > 0].nlargest(label_top, "coef"),
        ])
        if not chosen.empty:
            place_labels(ax, chosen["coef"], chosen["neglog_q"],
                         chosen["_label"], fontsize=6.0)

    ax.text(0.03, 0.96, f"{int(hit.sum())} of {len(df)} pairs at FDR < {alpha:g}",
            transform=ax.transAxes, va="top", fontsize=8, color="#444")
    add_direction_annotations(ax, supra_corner="lower left",
                              sub_corner="lower right")

    ax.set_xlabel(LBL_INTER, fontsize=9)
    ax.set_ylabel("$-\\log_{10}$ FDR", fontsize=9)
    ax.set_title("Enhancer pair interactions", fontweight="bold", fontsize=10)
    return ax


def plot_gene_interaction_heatmap(
    inters: pd.DataFrame,
    ax: plt.Axes | None = None,
    qval_col: str = "qval",
    alpha: float = 0.05,
    show_only_sig: bool = True,
) -> plt.Axes:
    """Heatmap of interaction coefficients, genes by enhancer pair.

    Args:
        inters: Interaction terms with gene, term and coef.
        ax: Axis to draw on; created when None.
        qval_col: Adjusted p-value column.
        alpha: Significance threshold.
        show_only_sig: Show significant cells only and drop empty rows and columns.

    Returns:
        The axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    if inters.empty:
        return empty_panel(ax, "No interaction terms", "Interaction map")

    values = inters.pivot_table(index="gene", columns="term", values="coef",
                                aggfunc="first")
    qcol = qval_col if qval_col in inters.columns else "pval"
    qvals = inters.pivot_table(index="gene", columns="term", values=qcol,
                               aggfunc="first")
    qvals = qvals.reindex(index=values.index, columns=values.columns)

    # Masking has to happen before the colour scale is set, otherwise a
    # discarded non-significant outlier would still fix vmax and wash out the
    # cells that remain.
    if show_only_sig:
        passes = qvals.to_numpy(dtype=float) < alpha
        values = pd.DataFrame(
            np.where(passes, values.to_numpy(dtype=float), np.nan),
            index=values.index, columns=values.columns,
        )
        values = values.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if values.empty:
            return empty_panel(ax, f"No pair at FDR < {alpha:g}",
                               "Interaction map")
        qvals = qvals.reindex(index=values.index, columns=values.columns)

    style_ax(ax)
    matrix = values.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1.0

    cmap = CMAP_INTERACTION.copy()
    cmap.set_bad("#E8E8E6")
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap,
                      vmin=-vmax, vmax=vmax, aspect="auto")

    qmatrix = qvals.to_numpy(dtype=float)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value, q = matrix[i, j], qmatrix[i, j]
            if not np.isfinite(value):
                continue
            if not show_only_sig and np.isfinite(q) and q >= alpha:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           hatch="///", edgecolor="#AAAAAA",
                                           linewidth=0, zorder=3))
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                    fontsize=7, color=PAL["ink"])

    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(values.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(values.index, fontsize=8.5)

    bar = plt.colorbar(image, ax=ax, shrink=0.7, pad=0.02)
    bar.set_label(f"$\\beta_{{ij}}$\n{LBL_DIRECTION}", fontsize=8)
    if show_only_sig:
        title = f"Interaction coefficients (FDR < {alpha:g} only)"
    else:
        title = f"Interaction coefficients (hatched: FDR $\\geq$ {alpha:g})"
    ax.set_title(title, fontweight="bold", fontsize=10)
    return ax


def plot_additivity(
    coop: pd.DataFrame,
    ax: plt.Axes | None = None,
    readout: str = "RNA",
    sum_col: str = "additive_prediction",
    all_col: str = "observed",
    name_col: str = "name",
    label_points: bool = True,
    label_top: int = 10,
) -> plt.Axes:
    """Observed all-scrambled effect against the additive prediction, one point per gene.

    Args:
        coop: build_coop_table output.
        ax: Axis to draw on; created when None.
        readout: Readout name for the axis labels.
        sum_col: Additive prediction column.
        all_col: All-scrambled effect column.
        name_col: Gene column.
        label_points: Annotate gene names.
        label_top: Number of genes labelled, ranked by absolute observed effect.

    Returns:
        The axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    if coop.empty or sum_col not in coop.columns:
        return empty_panel(ax, "No cooperativity table",
                           f"Additivity ({readout})")

    df = coop.dropna(subset=[sum_col, all_col])
    x = pd.to_numeric(df[sum_col], errors="coerce").to_numpy()
    y = pd.to_numeric(df[all_col], errors="coerce").to_numpy()
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3:
        return empty_panel(ax, "Too few genes", f"Additivity ({readout})")

    style_ax(ax)
    lim = np.array([min(x.min(), y.min()), max(x.max(), y.max())])
    pad = 0.08 * (lim[1] - lim[0] or 1.0)
    lim = lim + np.array([-pad, pad])

    ax.plot(lim, lim, color="gray", lw=1, ls="--", zorder=0)
    ax.fill_between(lim, lim, lim[1], color=PAL["sub"], alpha=0.05, zorder=0)
    ax.fill_between(lim, lim[0], lim, color=PAL["supra"], alpha=0.05, zorder=0)

    colors = np.where(y > x, PAL["sub"], PAL["supra"])
    ax.scatter(x, y, c=colors, s=65, edgecolors="white", lw=0.6, zorder=3)

    if label_points and label_top:
        names = df[name_col].to_numpy()[keep]
        chosen = np.argsort(-np.abs(y))[:label_top]
        place_labels(ax, x[chosen], y[chosen], names[chosen])

    r, p = stats.pearsonr(x, y)
    ax.text(0.04, 0.95, f"r = {r:.2f},  {pformat(p)}\nn = {len(x)} genes",
            transform=ax.transAxes, va="top", fontsize=8, color="#444")
    add_direction_annotations(ax)

    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel(f"Additive prediction  $\\Sigma_i\\, \\delta_i$   ({readout})",
                  fontsize=9)
    ax.set_ylabel(f"Observed  $\\delta_{{\\mathrm{{all}}}}$   ({readout})",
                  fontsize=9)
    ax.set_title(f"Deviation from independence ({readout})",
                 fontweight="bold", fontsize=10)
    return ax


def plot_epistasis_heatmap(
    epi: pd.DataFrame,
    ax: plt.Axes | None = None,
    a_col: str = "tfA",
    b_col: str = "tfB",
    value_col: str = "interaction",
    min_observations: int = 1,
) -> plt.Axes:
    """Heatmap of model-free pairwise interactions, enhancer by enhancer.

    Args:
        epi: build_pairwise_epistasis output.
        ax: Axis to draw on; created when None.
        a_col: First enhancer column.
        b_col: Second enhancer column.
        value_col: Interaction column.
        min_observations: Minimum observations to show a pair.

    Returns:
        The axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5.5))

    if epi.empty:
        return empty_panel(ax, "No pairwise interactions", "Pairwise epistasis")

    df = epi.copy()
    ordered = df.apply(lambda r: tuple(sorted([str(r[a_col]), str(r[b_col])])),
                       axis=1)
    df["_a"] = [p[0] for p in ordered]
    df["_b"] = [p[1] for p in ordered]

    agg = (df.groupby(["_a", "_b"])[value_col]
             .agg(mean="mean", n="size").reset_index())
    agg = agg[agg["n"] >= min_observations]
    if agg.empty:
        return empty_panel(ax, "No pair passes the occurrence filter",
                           "Pairwise epistasis")

    labels = sorted(set(agg["_a"]) | set(agg["_b"]))
    matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
    for _, row in agg.iterrows():
        matrix.loc[row["_a"], row["_b"]] = row["mean"]
        matrix.loc[row["_b"], row["_a"]] = row["mean"]

    style_ax(ax)
    values = matrix.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0
    cmap = CMAP_INTERACTION.copy()
    cmap.set_bad("#E8E8E6")
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap,
                      vmin=-vmax, vmax=vmax, aspect="equal")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.5)

    bar = plt.colorbar(image, ax=ax, shrink=0.6, pad=0.03)
    bar.set_label(f"Mean $I$\n{LBL_DIRECTION}", fontsize=8)
    ax.set_title("Pairwise epistasis", fontweight="bold", fontsize=10, y=1.03)
    return ax


def plot_interdependency(
    inter: pd.DataFrame,
    ax: plt.Axes | None = None,
    qval_col: str = "qval",
    alpha: float = 0.05,
    dim_nonsignificant: bool = True,
    annotate: bool = False,
) -> plt.Axes:
    """Heatmap of each enhancer perturbation's effect on accessibility at every locus of the gene.

    Args:
        inter: build_interdependency output.
        ax: Axis to draw on; created when None.
        qval_col: Adjusted p-value column (falls back to pval).
        alpha: Significance threshold.
        dim_nonsignificant: Fade non-significant cells.
        annotate: Print values in cells.

    Returns:
        The axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 8))

    if inter.empty:
        return empty_panel(ax, "No interdependency table", "Enhancer interdependency")

    # Order enhancers by gene so each gene forms a contiguous block.
    order, gene_sizes = [], {}
    for gene, group in inter.groupby("gene", sort=True):
        members = sorted(set(group["perturbed"]) | set(group["measured"]))
        order.extend(members)
        gene_sizes[gene] = len(members)

    index = {e: i for i, e in enumerate(order)}
    n = len(order)
    matrix = np.full((n, n), np.nan)
    qmatrix = np.full((n, n), np.nan)

    qcol = qval_col if qval_col in inter.columns else "pval"
    for _, row in inter.iterrows():
        i, j = index.get(row["perturbed"]), index.get(row["measured"])
        if i is None or j is None:
            continue
        matrix[i, j] = row["delta"]
        qmatrix[i, j] = row.get(qcol, np.nan)

    style_ax(ax)
    vmax = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1.0
    cmap = CMAP_INTERACTION.copy()
    cmap.set_bad("#E8E8E6")

    alphas = np.ones_like(matrix)
    if dim_nonsignificant:
        alphas[np.isfinite(qmatrix) & (qmatrix >= alpha)] = 0.25

    image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap,
                      vmin=-vmax, vmax=vmax, aspect="equal",
                      alpha=np.ma.masked_invalid(alphas).filled(1.0))

    if annotate:
        for i in range(n):
            for j in range(n):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center",
                            va="center", fontsize=6, color=PAL["ink"])

    short = [e.split("_", 1)[-1] if "_" in e else e for e in order]
    ax.set_xticks(range(n))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short, fontsize=7)
    ax.set_xlabel("Measured locus", fontsize=9)
    ax.set_ylabel("Perturbed enhancer", fontsize=9)

    cursor = 0
    for gene in sorted(gene_sizes):
        count = gene_sizes[gene]
        if cursor:
            ax.axhline(cursor - 0.5, color="white", lw=1.5, zorder=4)
            ax.axvline(cursor - 0.5, color="white", lw=1.5, zorder=4)
        ax.text(-0.04, cursor + count / 2 - 0.5, gene, ha="right", va="center",
                fontsize=7.5, fontweight="bold", color=PAL["ink"],
                transform=ax.get_yaxis_transform())
        cursor += count

    bar = plt.colorbar(image, ax=ax, shrink=0.4, pad=0.02)
    bar.set_label("$\\delta_i = \\log(y_i / y_0)$, ATAC", fontsize=8)
    ax.set_title(f"Enhancer interdependency (faded: FDR $\\geq$ {alpha:g})",
                 fontweight="bold", fontsize=10, pad=10)
    return ax


def _gene_blocks(labels: list[str], gene_of: dict[str, str]) -> list[tuple[str, int]]:
    """Group consecutive axis labels into (gene, run length) blocks."""
    blocks: list[list] = []
    for label in labels:
        gene = gene_of.get(label, "")
        if blocks and blocks[-1][0] == gene:
            blocks[-1][1] += 1
        else:
            blocks.append([gene, 1])
    return [(gene, count) for gene, count in blocks]


def plot_global_interdependency(
    atac_results: dict[str, dict],
    ax: plt.Axes | None = None,
    qval_col: str = "qval",
    alpha: float = 0.05,
    show_only_sig: bool = True,
    hatch_nonsignificant: bool = True,
    clip_quantile: float = 0.8,
    ylabel_pad: float = 58.0,
    gene_label_x: float = -0.095,
) -> plt.Axes:
    """Heatmap of the per-locus ATAC GLM main effects across all genes: perturbed enhancer by measured enhancer.

    Args:
        atac_results: Mapping of gene to ATAC GLM result with per_locus.
        ax: Axis to draw on; created when None.
        qval_col: Adjusted p-value column.
        alpha: Significance threshold.
        show_only_sig: Show significant cells only and drop empty rows and columns.
        hatch_nonsignificant: Hatch non-significant cells.
        clip_quantile: Quantile of |coefficient| setting the colour scale limits.
        ylabel_pad: Distance of the y label from the axis in points.
        gene_label_x: x position of the gene names in axes coordinates.

    Returns:
        The axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 8))

    gene_order, gene_enhancers, all_enhancers = [], {}, []
    for gene, result in atac_results.items():
        enhancers = result.get("enhancers")
        if not enhancers or not result.get("per_locus"):
            continue
        gene_order.append(gene)
        gene_enhancers[gene] = enhancers
        all_enhancers.extend(enhancers)

    if not all_enhancers:
        return empty_panel(ax, "No fitted ATAC models", "Global interdependency")

    index = {e: i for i, e in enumerate(all_enhancers)}
    n = len(all_enhancers)
    matrix = np.full((n, n), np.nan)
    qmatrix = np.full((n, n), np.nan)

    for gene in gene_order:
        per_locus = atac_results[gene]["per_locus"]
        for measured, table in per_locus.items():
            if table is None or measured not in index:
                continue
            for _, row in table.iterrows():
                term = str(row["term"])
                if ":" in term or "intercept" in term.lower():
                    continue
                perturbed = f"{gene}_{term}"
                if perturbed not in index:
                    continue
                matrix[index[perturbed], index[measured]] = row["coef"]
                qmatrix[index[perturbed], index[measured]] = row.get(
                    qval_col, row.get("pval", np.nan)
                )

    gene_of = {e: gene for gene in gene_order for e in gene_enhancers[gene]}
    row_labels, col_labels = list(all_enhancers), list(all_enhancers)

    # Filter before the colour scale is fixed, so a discarded cell cannot set
    # vmax and flatten the contrast of everything that survives.
    if show_only_sig:
        passes = np.isfinite(qmatrix) & (qmatrix < alpha)
        matrix = np.where(passes, matrix, np.nan)
        qmatrix = np.where(passes, qmatrix, np.nan)

        keep_rows = np.isfinite(matrix).any(axis=1)
        keep_cols = np.isfinite(matrix).any(axis=0)
        if not keep_rows.any():
            return empty_panel(ax, f"No cell at FDR < {alpha:g}",
                               "Global interdependency")

        matrix = matrix[np.ix_(keep_rows, keep_cols)]
        qmatrix = qmatrix[np.ix_(keep_rows, keep_cols)]
        row_labels = [e for e, keep in zip(all_enhancers, keep_rows) if keep]
        col_labels = [e for e, keep in zip(all_enhancers, keep_cols) if keep]

    style_ax(ax)
    finite = np.abs(matrix[np.isfinite(matrix)])
    if finite.size:
        vmax = float(np.quantile(finite, np.clip(clip_quantile, 0.0, 1.0)))
        # A degenerate quantile, which happens when most cells share a value,
        # would collapse the scale, so fall back to the full range.
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = float(finite.max()) or 1.0
    else:
        vmax = 1.0

    cmap = CMAP_INTERACTION.copy()
    cmap.set_bad(EMPTY_CELL)

    image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap,
                      vmin=-vmax, vmax=vmax, aspect="equal")

    if hatch_nonsignificant:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if not np.isfinite(matrix[i, j]):
                    continue
                q = qmatrix[i, j]
                if not np.isfinite(q) or q >= alpha:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                               fill=False, hatch="///",
                                               edgecolor="#666666",
                                               linewidth=0, zorder=4))

    def _short(name: str) -> str:
        return name.split("_", 1)[-1] if "_" in name else name

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels([_short(e) for e in col_labels], rotation=45,
                       ha="right", fontsize=7)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels([_short(e) for e in row_labels], fontsize=7)
    ax.set_xlabel("Measured locus", fontsize=9, fontweight="bold")
    ax.set_ylabel("Perturbed enhancer", fontsize=9, fontweight="bold",
                  labelpad=ylabel_pad)

    # Rows and columns can now hold different genes, so the two sets of
    # separators are derived independently.
    cursor = 0
    for gene, count in _gene_blocks(row_labels, gene_of):
        if cursor:
            ax.axhline(cursor - 0.5, color="white", lw=1.5, zorder=5)
        ax.text(gene_label_x, cursor + count / 2 - 0.5, gene, ha="right",
                va="center", fontsize=7.5, fontweight="bold", color=PAL["ink"],
                transform=ax.get_yaxis_transform())
        cursor += count

    cursor = 0
    for _, count in _gene_blocks(col_labels, gene_of):
        if cursor:
            ax.axvline(cursor - 0.5, color="white", lw=1.5, zorder=5)
        cursor += count

    clipped = clip_quantile < 1.0 and np.isfinite(matrix).any()
    bar = plt.colorbar(image, ax=ax, shrink=0.4, pad=0.02,
                       extend="both" if clipped else "neither")
    label = "$\\beta_i$, ATAC (log scale)"
    if clipped:
        label += f"\nscale clipped at $\\pm${vmax:.2f}"
    bar.set_label(label, fontsize=8)
    if show_only_sig:
        title = f"Enhancer interdependency (FDR < {alpha:g} only)"
    elif hatch_nonsignificant:
        title = f"Enhancer interdependency (hatched: FDR $\\geq$ {alpha:g})"
    else:
        title = "Enhancer interdependency"
    ax.set_title(title, fontweight="bold", fontsize=10, pad=10)
    return ax


# --------------------------------------------------------------------------- #
# Validation against the measured screen
# --------------------------------------------------------------------------- #

def _split_locus(name: str) -> tuple[str, str]:
    """Split "GENE_enhN" into gene and enhancer."""
    text = str(name)
    if "_" in text:
        gene, enhancer = text.split("_", 1)
        return gene, enhancer
    return "", text


def _locus_labels(df: pd.DataFrame, level: str) -> pd.Series:
    """Build two-line point labels (gene over enhancer or enhancer pair) for a comparison table."""
    if "label" in df.columns:
        return df["label"].astype(str)

    if {"gene", "term"}.issubset(df.columns):
        return df["gene"].astype(str) + "\n" + df["term"].astype(str)

    for a_col, b_col in (("full_a", "full_b"), ("enhA", "enhB")):
        if {a_col, b_col}.issubset(df.columns):
            labels = []
            for a, b in zip(df[a_col], df[b_col]):
                gene_a, enh_a = _split_locus(a)
                gene_b, enh_b = _split_locus(b)
                gene = gene_a or gene_b
                if level == "single":
                    labels.append(f"{gene}\n{enh_a}" if gene else enh_a)
                else:
                    labels.append(f"{gene}\n{enh_a}:{enh_b}" if gene
                                  else f"{enh_a}:{enh_b}")
            return pd.Series(labels, index=df.index)

    for single_col in ("full_a", "enhancer", "locus", "name", "pair_name"):
        if single_col in df.columns:
            labels = []
            for value in df[single_col]:
                gene, enhancer = _split_locus(value)
                labels.append(f"{gene}\n{enhancer}" if gene else enhancer)
            return pd.Series(labels, index=df.index)

    return df.index.astype(str).to_series(index=df.index)


def plot_insilico_vs_measured(
    comp: pd.DataFrame,
    ax: plt.Axes | None = None,
    insilico_col: str = "coef",
    measured_col: str = "interaction_beta",
    level: str = "single",
    color_by_direction: bool = True,
    equal_axes: bool = False,
    label_points: bool = True,
    label_top: int = 5,
    label_col: str | None = None,
) -> plt.Axes:
    """Scatter of in silico against measured CRISPRi coefficients with fit line and statistics.

    Args:
        comp: Merged comparison table.
        ax: Axis to draw on; created when None.
        insilico_col: In silico coefficient column.
        measured_col: Measured coefficient column.
        level: "single", "pair" or a free title label.
        color_by_direction: Colour interaction points by sign (ignored for singles).
        equal_axes: Use a shared square range with an identity line.
        label_points: Annotate points.
        label_top: Number of points labelled, ranked by |in silico coefficient|.
        label_col: Label column; built from the enhancer columns when None.

    Returns:
        The axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    if comp.empty:
        return empty_panel(ax, "No overlapping measurements",
                           f"In silico versus CRISPRi ({level})")

    df = comp.dropna(subset=[insilico_col, measured_col])
    if len(df) < 3:
        return empty_panel(ax, "Too few overlapping measurements",
                           f"In silico versus CRISPRi ({level})")

    style_ax(ax)
    x = df[measured_col].to_numpy()
    y = df[insilico_col].to_numpy()

    if color_by_direction and level != "single":
        colors = np.where(y < 0, PAL["supra"], PAL["sub"])
    else:
        colors = PAL["atac"]

    ax.axhline(0, color="gray", lw=0.8, ls=":", zorder=0)
    ax.axvline(0, color="gray", lw=0.8, ls=":", zorder=0)

    if equal_axes:
        span = np.array([min(x.min(), y.min()), max(x.max(), y.max())])
        pad = 0.1 * (span[1] - span[0] or 1.0)
        span = span + np.array([-pad, pad])
        ax.plot(span, span, color="gray", lw=1, ls="--", zorder=0)
        ax.set_xlim(*span)
        ax.set_ylim(*span)

    ax.scatter(x, y, c=colors, s=70, alpha=0.85, edgecolors="white",
               lw=0.6, zorder=3)

    # Rank on the in silico estimate, so the annotated points are the ones the
    # model itself calls strongest rather than the ones the screen calls
    # strongest.
    if label_points and label_top:
        if label_col and label_col in df.columns:
            labels = df[label_col].astype(str)
        else:
            labels = _locus_labels(df, level)
        chosen = np.argsort(-np.abs(y))[:label_top]
        place_labels(ax, x[chosen], y[chosen],
                     labels.to_numpy()[chosen], fontsize=6.0)

    slope, intercept, r, p_r, _ = stats.linregress(x, y)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, intercept + slope * xs, color=PAL["ink"], lw=1.1, ls=":",
            zorder=2)

    rho, p_rho = stats.spearmanr(x, y)
    concordant = float((np.sign(x) == np.sign(y)).mean())

    ax.text(0.04, 0.96,
            f"Spearman $\\rho$ = {rho:.2f}, {pformat(p_rho)}\n"
            f"Pearson r = {r:.2f}\n"
            f"slope = {slope:.2f}  (1 would be calibrated)\n"
            f"{concordant:.0%} sign concordant,  n = {len(df)}",
            transform=ax.transAxes, va="top", fontsize=7.5, color="#444")

    label = "$\\beta_{ij}$" if level != "single" else "$\\beta_i$"
    ax.set_xlabel(f"Measured CRISPRi  {label}", fontsize=9)
    ax.set_ylabel(f"In silico  {label}", fontsize=9)
    ax.set_title(f"In silico versus CRISPRi ({level})",
                 fontweight="bold", fontsize=10)
    return ax


def plot_validation_summary(
    summaries: dict[str, dict],
    ax: plt.Axes | None = None,
    subsets: tuple[str, ...] = ("", "_insilico"),
    labels: tuple[str, ...] = ("all effects", "significant in silico"),
) -> plt.Axes:
    """Bar chart of sign concordance with the screen, with Wilson intervals, per level and subset.

    Args:
        summaries: Mapping of level to summarise_validation output.
        ax: Axis to draw on; created when None.
        subsets: Key suffixes to draw.
        labels: Legend labels matching subsets.

    Returns:
        The axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4))

    usable = {k: v for k, v in summaries.items() if v and v.get("n")}
    if not usable:
        return empty_panel(ax, "No validation summary", "Validation")

    style_ax(ax)
    levels = list(usable)
    colors = [PAL["atac"], PAL["crispri"], PAL["supra"]][:len(subsets)]
    height = 0.8 / len(subsets)
    offsets = np.linspace((len(subsets) - 1) / 2, -(len(subsets) - 1) / 2,
                          len(subsets)) * height

    for k, (suffix, label, color, offset) in enumerate(
            zip(subsets, labels, colors, offsets)):
        values, lows, highs, positions, annotations = [], [], [], [], []
        for i, level in enumerate(levels):
            summary = usable[level]
            rate = summary.get(f"concordance{suffix}", np.nan)
            if not np.isfinite(rate):
                continue
            values.append(rate)
            lows.append(rate - summary.get(f"ci_lo{suffix}", rate))
            highs.append(summary.get(f"ci_hi{suffix}", rate) - rate)
            positions.append(i + offset)
            annotations.append(
                f"{summary.get(f'n_concordant{suffix}', 0)}/"
                f"{summary.get(f'n{suffix}', 0)}")

        if not values:
            continue
        ax.barh(positions, values, height=height * 0.9, color=color,
                edgecolor="white", lw=0.5, label=label, zorder=2)
        ax.errorbar(values, positions, xerr=[lows, highs], fmt="none",
                    ecolor="#444", elinewidth=1.1, capsize=2.5, zorder=3)
        for value, position, text in zip(values, positions, annotations):
            ax.text(min(value + max(highs) + 0.03, 1.14), position, text,
                    va="center", fontsize=6.8, color="#555")

    ax.axvline(0.5, color="gray", lw=1.2, ls="--", zorder=1)
    ax.text(0.5, len(levels) - 0.42, "chance", ha="center", va="bottom",
            fontsize=7, color="gray")

    ax.set_yticks(np.arange(len(levels)))
    ax.set_yticklabels(levels, fontsize=9)
    ax.set_xlim(0, 1.2)
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.set_xlabel("Sign concordance with CRISPRi\n"
                  "(bars: Wilson 95% CI)", fontsize=9)
    ax.legend(fontsize=6.5, loc="lower right", framealpha=0.9)
    ax.set_title("Validation against the measured screen",
                 fontweight="bold", fontsize=10)
    return ax


def plot_trans_vs_cooperativity(
    trans: pd.DataFrame,
    coop: pd.DataFrame,
    ax: plt.Axes | None = None,
    trans_key: str = "gene",
    coop_key: str = "name",
    trans_col: str = "mean_trans",
    coop_col: str = "cooperativity",
    readout: str = "ATAC",
    resolution: str = "per gene",
    label_points: bool = False,
) -> plt.Axes:
    """Mean trans accessibility effect against cooperativity.

    Args:
        trans: summarise_trans_effects output.
        coop: Cooperativity table at the same resolution.
        ax: Axis to draw on; created when None.
        trans_key: Join column in trans.
        coop_key: Join column in coop.
        trans_col: Mean trans effect column.
        coop_col: Cooperativity column.
        readout: Readout name for the y label.
        resolution: Grouping description for the title.
        label_points: Annotate group names.

    Returns:
        The axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.2))

    if trans is None or trans.empty or coop is None or coop.empty:
        return empty_panel(ax, "Input missing", "Trans vs cooperativity")

    merged = trans.merge(coop, left_on=trans_key, right_on=coop_key,
                         how="inner").dropna(subset=[trans_col, coop_col])
    if len(merged) < 3:
        return empty_panel(ax, f"Only {len(merged)} matched group(s)",
                           "Trans vs cooperativity")

    style_ax(ax)
    x = merged[trans_col].to_numpy()
    y = merged[coop_col].to_numpy()

    ax.axhline(0, color="gray", lw=0.8, ls="--", zorder=0)
    ax.axvline(0, color="gray", lw=0.8, ls="--", zorder=0)
    colors = [PAL["supra"] if v < 0 else PAL["sub"] for v in y]
    ax.scatter(x, y, c=colors, s=55, alpha=0.85, edgecolors="white",
               lw=0.6, zorder=3)

    if label_points:
        place_labels(ax, x, y, merged[trans_key])

    lines = []
    if len(np.unique(x)) > 1 and len(np.unique(y)) > 1:
        slope, intercept, r, p, _ = stats.linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, intercept + slope * xs, color=PAL["ink"], lw=1.2, zorder=2)
        rho, p_rho = stats.spearmanr(x, y)
        lines = [f"r = {r:+.3f}  {pformat(p)}",
                 f"rho = {rho:+.3f}  {pformat(p_rho)}",
                 f"n = {len(merged)}"]
    ax.text(0.04, 0.06, "\n".join(lines) or "not estimable",
            transform=ax.transAxes, va="bottom", fontsize=7.5, color="#444",
            family="monospace")

    ax.set_xlabel("Mean trans effect on accessibility  $\\bar{\\delta}_{trans}$",
                  fontsize=9)
    ax.set_ylabel(f"Cooperativity $I$   ({readout})", fontsize=9)
    ax.set_title(f"Trans effect vs cooperativity ({resolution})",
                 fontweight="bold", fontsize=10)
    return ax


def plot_pair_trans_vs_interaction(
    comp: pd.DataFrame,
    ax: plt.Axes | None = None,
    trans_col: str = "trans_strongest",
    value_col: str = "interaction_beta",
    value_label: str = "Measured CRISPRi  $\\beta_{ij}$",
    trans_label: str | None = None,
    label_points: bool = False,
) -> plt.Axes:
    """Interaction coefficient of each enhancer pair against its trans accessibility statistic.

    Args:
        comp: Comparison table after join_pair_trans.
        ax: Axis to draw on; created when None.
        trans_col: Trans statistic for x, e.g. trans_strongest or trans_asymmetry.
        value_col: Interaction coefficient for y (measured or in silico).
        value_label: y-axis label.
        trans_label: x-axis label; derived when None.
        label_points: Annotate gene names.

    Returns:
        The axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.2))

    if comp is None or comp.empty or trans_col not in comp.columns:
        return empty_panel(ax, "Run enhancer_analysis.join_pair_trans",
                           "Crosstalk vs interaction")

    df = comp.dropna(subset=[trans_col, value_col])
    if len(df) < 3:
        return empty_panel(ax, f"Only {len(df)} matched pair(s)",
                           "Crosstalk vs interaction")

    style_ax(ax)
    x = df[trans_col].to_numpy()
    y = df[value_col].to_numpy()

    ax.axhline(0, color="gray", lw=0.8, ls="--", zorder=0)
    colors = [PAL["supra"] if v < 0 else PAL["sub"] for v in y]
    ax.scatter(x, y, c=colors, s=55, alpha=0.85, edgecolors="white",
               lw=0.6, zorder=3)

    if label_points and "gene" in df.columns:
        place_labels(ax, x, y, df["gene"])

    lines = []
    if len(np.unique(x)) > 1 and len(np.unique(y)) > 1:
        slope, intercept, r, p, _ = stats.linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, intercept + slope * xs, color=PAL["ink"], lw=1.2, zorder=2)
        rho, p_rho = stats.spearmanr(x, y)
        lines = [f"r = {r:+.3f}  {pformat(p)}",
                 f"rho = {rho:+.3f}  {pformat(p_rho)}",
                 f"n = {len(df)}"]
    ax.text(0.04, 0.06, "\n".join(lines) or "not estimable",
            transform=ax.transAxes, va="bottom", fontsize=7.5, color="#444",
            family="monospace")

    defaults = {
        "trans_strongest": "Strongest trans effect between the pair",
        "trans_asymmetry": "Directional asymmetry of the trans effect",
        "trans_mean": "Mean trans effect between the pair",
    }
    ax.set_xlabel(trans_label or defaults.get(trans_col, trans_col), fontsize=9)
    ax.set_ylabel(value_label, fontsize=9)
    ax.set_title("Accessibility crosstalk vs interaction",
                 fontweight="bold", fontsize=10)
    return ax


def make_mechanism_figure(
    trans_by_gene: pd.DataFrame,
    trans_by_locus: pd.DataFrame,
    coop_atac_gene: pd.DataFrame,
    coop_atac_locus: pd.DataFrame,
    coop_rna_gene: pd.DataFrame,
    *,
    pair_comp: pd.DataFrame | None = None,
    figsize: tuple[float, float] = (8, 14),
) -> plt.Figure:
    """Assemble the figure relating trans accessibility crosstalk to cooperativity and pair interactions.

    Args:
        trans_by_gene: Trans effects per gene.
        trans_by_locus: Trans effects per perturbed enhancer.
        coop_atac_gene: Gene-level cooperativity on ATAC.
        coop_atac_locus: Locus-level cooperativity on ATAC.
        coop_rna_gene: Gene-level cooperativity on RNA.
        pair_comp: Comparison table after join_pair_trans; placeholders when None.
        figsize: Figure size in inches.

    Returns:
        The figure.
    """
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    fig.patch.set_facecolor("#FAFAF8")
    flat = axes.ravel()

    plot_trans_vs_cooperativity(trans_by_gene, coop_atac_gene, ax=flat[4],
                                trans_key="gene", readout="ATAC",
                                resolution="per gene")
    plot_trans_vs_cooperativity(trans_by_locus, coop_atac_locus, ax=flat[0],
                                trans_key="perturbed", readout="ATAC",
                                resolution="per locus")
    plot_trans_vs_cooperativity(trans_by_gene, coop_rna_gene, ax=flat[5],
                                trans_key="gene", readout="RNA",
                                resolution="per gene")

    blank = pd.DataFrame()
    comp = pair_comp if pair_comp is not None else blank
    plot_pair_trans_vs_interaction(
        comp, ax=flat[1], trans_col="trans_strongest",
        value_col="interaction_beta",
        value_label="Measured CRISPRi  $\\beta_{ij}$")
    plot_pair_trans_vs_interaction(
        comp, ax=flat[2], trans_col="trans_strongest", value_col="coef",
        value_label="In silico  $\\beta_{ij}$")
    plot_pair_trans_vs_interaction(
        comp, ax=flat[3], trans_col="trans_asymmetry",
        value_col="interaction_beta",
        value_label="Measured CRISPRi  $\\beta_{ij}$")

    panel_letters(flat, dx=-0.18)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.suptitle("Does inter-enhancer chromatin crosstalk predict cooperativity?",
                 fontsize=12, fontweight="bold", y=0.985)
    return fig


def plot_atac_rna_interaction_concordance(
    rna_inters: pd.DataFrame,
    atac_inters: pd.DataFrame,
    ax: plt.Axes | None = None,
    key_cols: tuple[str, str] = ("gene", "term"),
) -> plt.Axes:
    """Interaction coefficients on expression against accessibility for the same terms.

    Args:
        rna_inters: Interactions fitted on RNA.
        atac_inters: Interactions fitted on ATAC.
        ax: Axis to draw on; created when None.
        key_cols: Columns identifying the same term in both tables.

    Returns:
        The axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))

    if rna_inters.empty or atac_inters.empty:
        return empty_panel(ax, "One readout is missing", "RNA versus ATAC")

    merged = rna_inters.merge(atac_inters, on=list(key_cols),
                              suffixes=("_rna", "_atac"))
    merged = merged.dropna(subset=["coef_rna", "coef_atac"])
    if len(merged) < 3:
        return empty_panel(ax, "Too few shared terms", "RNA versus ATAC")

    style_ax(ax)
    x = merged["coef_atac"].to_numpy()
    y = merged["coef_rna"].to_numpy()

    ax.axhline(0, color="gray", lw=0.8, ls="--", zorder=0)
    ax.axvline(0, color="gray", lw=0.8, ls="--", zorder=0)

    concordant = np.sign(x) == np.sign(y)
    colors = np.where(concordant,
                      np.where(x < 0, PAL["supra"], PAL["sub"]), PAL["neut"])
    ax.scatter(x, y, c=colors, s=60, edgecolors="white", lw=0.6, zorder=3)

    rho, p = stats.spearmanr(x, y)
    ax.text(0.04, 0.96,
            f"Spearman $\\rho$ = {rho:.2f}, {pformat(p)}\n"
            f"{int(concordant.sum())}/{len(merged)} concordant in sign",
            transform=ax.transAxes, va="top", fontsize=8, color="#444")

    ax.set_xlabel("$\\beta_{ij}$, ATAC", fontsize=9)
    ax.set_ylabel("$\\beta_{ij}$, RNA", fontsize=9)
    ax.set_title("Interaction: accessibility versus expression",
                 fontweight="bold", fontsize=10)
    return ax


# --------------------------------------------------------------------------- #
# Artefact and mechanism checks
# --------------------------------------------------------------------------- #

def plot_saturation_diagnostic(
    pairs: pd.DataFrame,
    ax: plt.Axes | None = None,
    against: str = "additive",
    coef_col: str = "coef",
    main_a_col: str = "main_a",
    main_b_col: str = "main_b",
    baseline_col: str = "baseline",
    qval_col: str = "qval",
    alpha: float = 0.05,
) -> plt.Axes:
    """Interaction against the additive prediction or the baseline, to check for floor and ceiling artefacts.

    Args:
        pairs: Interactions after attach_main_effects.
        ax: Axis to draw on; created when None.
        against: "additive" or "baseline".
        coef_col: Interaction column.
        main_a_col: First main effect column.
        main_b_col: Second main effect column.
        baseline_col: Baseline readout column.
        qval_col: Adjusted p-value column, for colouring.
        alpha: Significance threshold, for colouring.

    Returns:
        The axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4))

    if against == "additive":
        if not {main_a_col, main_b_col}.issubset(pairs.columns):
            return empty_panel(
                ax,
                "Main effects missing\nrun enhancer_analysis.attach_main_effects",
                "Saturation check (additive)")
        x = (pairs[main_a_col] + pairs[main_b_col]).to_numpy()
        xlabel = "Additive prediction  $\\beta_i + \\beta_j$"
        note = "floor artefact would slope downwards"
    elif against == "baseline":
        if baseline_col not in pairs.columns:
            return empty_panel(
                ax,
                "Baseline missing\nrun enhancer_analysis.attach_main_effects",
                "Saturation check (baseline)")
        x = pairs[baseline_col].to_numpy()
        xlabel = "Unperturbed readout  $y_0$"
        note = "ceiling artefact would slope downwards"
    else:
        raise ValueError(f"Unknown diagnostic '{against}'.")

    y = pairs[coef_col].to_numpy()
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3:
        return empty_panel(ax, "Too few pairs", "Saturation check")

    style_ax(ax)
    if qval_col in pairs.columns:
        colors = np.where(pairs.loc[keep, qval_col].to_numpy() < alpha,
                          PAL["atac"], PAL["neut"])
    else:
        colors = PAL["atac"]

    ax.scatter(x, y, c=colors, s=42, alpha=0.8, edgecolors="white",
               lw=0.5, zorder=3)
    ax.axhline(0, color="gray", lw=0.8, ls="--", zorder=0)

    # A constant x carries no trend and would crash the regression.
    if len(np.unique(x)) < 2:
        ax.text(0.04, 0.05,
                f"{xlabel.split('  ')[0]} is constant\nno trend estimable",
                transform=ax.transAxes, va="bottom", fontsize=7.5, color="#444")
    else:
        slope, intercept, r, p, _ = stats.linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, intercept + slope * xs, color=PAL["ink"], lw=1.2, zorder=2)
        ax.text(0.04, 0.05, f"r = {r:.2f},  {pformat(p)}\n{note}",
                transform=ax.transAxes, va="bottom", fontsize=7.5, color="#444")

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(LBL_INTER, fontsize=9)
    ax.set_title(f"Saturation check ({against})", fontweight="bold", fontsize=10)
    return ax


def plot_interaction_vs_distance(
    pairs: pd.DataFrame,
    ax: plt.Axes | None = None,
    coef_col: str = "coef",
    distance_col: str = "distance_bp",
    qval_col: str = "qval",
    alpha: float = 0.05,
    log_x: bool = True,
) -> plt.Axes:
    """Interaction against the genomic distance between the two enhancers.

    Args:
        pairs: Interactions with distance_bp.
        ax: Axis to draw on; created when None.
        coef_col: Interaction column.
        distance_col: Distance column (bp).
        qval_col: Adjusted p-value column, for colouring.
        alpha: Significance threshold, for colouring.
        log_x: Logarithmic distance axis.

    Returns:
        The axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    if pairs.empty or distance_col not in pairs.columns:
        return empty_panel(ax, "No distance annotation",
                           "Interaction versus distance")

    df = pairs.dropna(subset=[distance_col, coef_col])
    df = df[df[distance_col] > 0]
    if len(df) < 3:
        return empty_panel(ax, "Too few pairs", "Interaction versus distance")

    style_ax(ax)
    if qval_col in df.columns:
        colors = np.where(df[qval_col] < alpha, PAL["atac"], PAL["neut"])
    else:
        colors = PAL["atac"]

    ax.scatter(df[distance_col], df[coef_col], c=colors, s=45, alpha=0.8,
               edgecolors="white", lw=0.5, zorder=3)
    ax.axhline(0, color="gray", lw=0.8, ls="--", zorder=0)
    if log_x:
        ax.set_xscale("log")

    rho, p = stats.spearmanr(df[distance_col], df[coef_col])
    ax.text(0.04, 0.05,
            f"Spearman $\\rho$ = {rho:.2f},  {pformat(p)}\nn = {len(df)} pairs",
            transform=ax.transAxes, va="bottom", fontsize=8, color="#444")

    ax.set_xlabel("Distance between enhancer midpoints (bp)", fontsize=9)
    ax.set_ylabel(LBL_INTER, fontsize=9)
    ax.set_title("Distance dependence", fontweight="bold", fontsize=10)
    return ax


# --------------------------------------------------------------------------- #
# Cross-level synthesis
# --------------------------------------------------------------------------- #

def plot_cross_level(
    tf_coop: pd.DataFrame,
    enhancer_mains: pd.DataFrame,
    ax: plt.Axes | None = None,
    tf_enhancer_col: str = "enhancer",
    tf_value_col: str = "cooperativity",
    enh_name_col: str = "full_a",
    enh_value_col: str = "coef",
    label_points: bool = True,
) -> plt.Axes:
    """Within-enhancer motif cooperativity against the enhancer's own main effect.

    Args:
        tf_coop: Per-enhancer cooperativity from the TF analysis.
        enhancer_mains: Enhancer-level main effects.
        ax: Axis to draw on; created when None.
        tf_enhancer_col: Enhancer column in tf_coop.
        tf_value_col: Cooperativity column in tf_coop.
        enh_name_col: Enhancer column in enhancer_mains.
        enh_value_col: Coefficient column in enhancer_mains.
        label_points: Annotate enhancer names.

    Returns:
        The axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5))

    if tf_coop.empty or enhancer_mains.empty:
        return empty_panel(ax, "One level is missing", "Motif grammar versus effect")

    merged = tf_coop.merge(
        enhancer_mains[[enh_name_col, enh_value_col]],
        left_on=tf_enhancer_col, right_on=enh_name_col, how="inner",
    ).dropna(subset=[tf_value_col, enh_value_col])

    if len(merged) < 3:
        return empty_panel(ax, "Too few enhancers in both analyses",
                           "Motif grammar versus effect")

    style_ax(ax)
    x = merged[tf_value_col].to_numpy()
    y = merged[enh_value_col].to_numpy()

    ax.axhline(0, color="gray", lw=0.8, ls="--", zorder=0)
    ax.axvline(0, color="gray", lw=0.8, ls="--", zorder=0)
    colors = [PAL["supra"] if v < 0 else PAL["sub"] for v in x]
    ax.scatter(x, y, c=colors, s=60, edgecolors="white", lw=0.6, zorder=3)

    if label_points:
        place_labels(ax, x, y, merged[tf_enhancer_col])

    rho, p = stats.spearmanr(x, y)
    ax.text(0.04, 0.96,
            f"Spearman $\\rho$ = {rho:.2f},  {pformat(p)}\n"
            f"n = {len(merged)} enhancers",
            transform=ax.transAxes, va="top", fontsize=8, color="#444")

    ax.set_xlabel(f"Internal motif cooperativity  $I$\n{LBL_DIRECTION}",
                  fontsize=9)
    ax.set_ylabel("Enhancer main effect  $\\beta_i$   (RNA)", fontsize=9)
    ax.set_title("Motif grammar versus enhancer effect",
                 fontweight="bold", fontsize=10)
    return ax


def _as_list(value: Any) -> list:
    """Convert a list-like cell (list, array or its string form) into a list."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return list(parsed) if isinstance(parsed, (list, tuple)) else [value]
        except (ValueError, SyntaxError):
            return [value]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [value]


def _as_float(value: Any) -> float:
    """Convert a scalar or single-element sequence to float."""
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=float).ravel()
        return float(arr.mean()) if arr.size else np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def plot_upset(
    scram_agg: pd.DataFrame,
    gene: str,
    ax_bar: plt.Axes,
    ax_dots: plt.Axes,
    value_col: str = "RNA_counts",
    sd_col: str = "RNA_counts_sd",
    name_col: str = "name",
    scrambled_col: str = "scrambled_seqlets",
    enhancer_col: str = "target_enhancer",
    n_scrambled_col: str = "n_scrambled",
    normalise: bool = True,
    dot_size: float = 90,
) -> tuple[plt.Axes, plt.Axes]:
    """UpSet-style plot of every scramble combination of one gene: readout bars over a membership matrix.

    Args:
        scram_agg: Shuffle-averaged scramble table, one row per combination.
        gene: Gene to show.
        ax_bar: Axis for the bars.
        ax_dots: Axis for the membership matrix.
        value_col: Readout column.
        sd_col: Standard deviation column across shuffles.
        name_col: Gene column.
        scrambled_col: Column with the scrambled enhancers.
        enhancer_col: Column with the ordered enhancer list.
        n_scrambled_col: Column with the number scrambled.
        normalise: Plot the log fold change against the reference instead of raw values.
        dot_size: Marker size in the matrix.

    Returns:
        Tuple of (bar axis, matrix axis).
    """
    group = scram_agg[scram_agg[name_col] == gene].copy()
    if group.empty:
        empty_panel(ax_bar, f"No rows for {gene}", gene)
        ax_dots.axis("off")
        return ax_bar, ax_dots

    group["_scr"] = group[scrambled_col].apply(_as_list)
    group["_value"] = group[value_col].apply(_as_float)
    group["_sd"] = (group[sd_col].apply(_as_float)
                    if sd_col in group.columns else 0.0)
    group["_n"] = group[n_scrambled_col].astype(int)
    group = group.sort_values(["_n", "_scr"], key=lambda s: s.map(str))

    baseline_rows = group[group["_n"] == 0]
    if baseline_rows.empty:
        empty_panel(ax_bar, f"No unperturbed row for {gene}", gene)
        ax_dots.axis("off")
        return ax_bar, ax_dots

    baseline = baseline_rows.iloc[0]
    enhancers = _as_list(baseline[enhancer_col]) or sorted(
        {e for lst in group["_scr"] for e in lst}
    )
    short = [e.split("_", 1)[-1] if "_" in e else e for e in enhancers]
    base_value = baseline["_value"]

    if normalise and np.isfinite(base_value) and base_value > 0:
        values = np.log2(group["_value"].to_numpy() / base_value)
        errors = group["_sd"].to_numpy() / (group["_value"].to_numpy() * np.log(2))
        ylabel = "$\\log_2$ FC versus unperturbed"
        reference = 0.0
    else:
        values = group["_value"].to_numpy()
        errors = group["_sd"].to_numpy()
        ylabel = value_col
        reference = base_value

    x = np.arange(len(group))

    style_ax(ax_bar)
    ax_bar.spines["bottom"].set_visible(False)
    ax_bar.tick_params(bottom=False, labelbottom=False)

    is_baseline = group["_n"].to_numpy() == 0
    colors = [PAL["rna"] if b else PAL["atac"] for b in is_baseline]
    ax_bar.bar(x, values, color=colors, width=0.65, edgecolor="white",
               linewidth=0.5, zorder=2)

    finite_errors = np.where(np.isfinite(errors), errors, 0)
    if np.any(finite_errors > 0):
        ax_bar.errorbar(x, values, yerr=finite_errors, fmt="none", color="#444",
                        lw=1.1, capsize=2.5, zorder=3)

    ax_bar.axhline(reference, color=PAL["rna"], lw=1.2, ls="--", alpha=0.7,
                   zorder=1)
    ax_bar.set_ylabel(ylabel, fontsize=9)
    ax_bar.set_xlim(-0.7, len(group) - 0.3)
    ax_bar.set_title(f"{gene}: every scramble combination",
                     fontweight="bold", fontsize=10)

    ax_dots.set_facecolor(PAL["bg"])
    ax_dots.spines[:].set_visible(False)
    ax_dots.tick_params(left=False, bottom=False, labelbottom=False)

    for row_idx, enhancer in enumerate(enhancers):
        y = len(enhancers) - 1 - row_idx
        member = [enhancer in scr for scr in group["_scr"]]
        ax_dots.scatter(x, np.full(len(x), y), s=dot_size,
                        c=[PAL["ink"] if m else "#DDDDDD" for m in member],
                        zorder=3)

    # Connect the scrambled enhancers within each combination.
    for column, scr in zip(x, group["_scr"]):
        rows = [len(enhancers) - 1 - enhancers.index(e)
                for e in scr if e in enhancers]
        if len(rows) > 1:
            ax_dots.plot([column, column], [min(rows), max(rows)],
                         color=PAL["ink"], lw=1.6, zorder=2)

    ax_dots.set_yticks(range(len(enhancers)))
    ax_dots.set_yticklabels(short[::-1], fontsize=8)
    ax_dots.tick_params(left=True, labelleft=True)
    ax_dots.set_xlim(-0.7, len(group) - 0.3)
    ax_dots.set_ylim(-0.6, len(enhancers) - 0.4)
    return ax_bar, ax_dots


def make_gene_figure(
    gene: str,
    rna_result: dict,
    epi: pd.DataFrame,
    scram_agg: pd.DataFrame,
    *,
    qval_col: str = "qval",
    alpha: float = 0.05,
    figsize: tuple[float, float] = (13, 11),
    **upset_kwargs: Any,
) -> plt.Figure:
    """Assemble a single-gene figure: coefficient forest, pairwise epistasis and UpSet plot.

    Args:
        gene: Gene to show.
        rna_result: fit_all_genes result of the gene.
        epi: build_pairwise_epistasis output.
        scram_agg: Shuffle-averaged scramble table.
        qval_col: Adjusted p-value column.
        alpha: Significance threshold.
        figsize: Figure size in inches.
        **upset_kwargs: Passed to plot_upset.

    Returns:
        The figure.
    """
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("#FAFAF8")
    grid = GridSpec(3, 2, figure=fig, height_ratios=[3.0, 1.9, 1.1],
                    hspace=0.35, wspace=0.28,
                    left=0.09, right=0.96, top=0.91, bottom=0.05)

    ax_forest = fig.add_subplot(grid[0, 0])
    ax_epi = fig.add_subplot(grid[0, 1])
    ax_bar = fig.add_subplot(grid[1, :])
    ax_dots = fig.add_subplot(grid[2, :])

    plot_coef_forest(rna_result, ax=ax_forest, qval_col=qval_col, alpha=alpha)

    # Crop the epistasis table to this gene only.
    gene_epi = epi[epi["enhancer"] == gene] if "enhancer" in epi.columns \
        else epi.iloc[0:0]
    if gene_epi.empty:
        empty_panel(ax_epi, f"No pairwise terms for {gene}", "Pairwise epistasis")
    else:
        plot_epistasis_heatmap(gene_epi, ax=ax_epi)

    plot_upset(scram_agg, gene, ax_bar, ax_dots, **upset_kwargs)

    for ax, letter in zip([ax_forest, ax_epi, ax_bar], "ABC"):
        ax.text(-0.14, 1.04, letter, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="bottom")

    fig.suptitle(f"Worked example: {gene}", fontsize=13, fontweight="bold",
                 y=0.955)
    return fig


def plot_coop_agreement(
    coop_a: pd.DataFrame,
    coop_b: pd.DataFrame,
    ax: plt.Axes | None = None,
    labels: tuple[str, str] = ("summed", "per locus"),
    name_col: str = "name",
    value_col: str = "cooperativity",
    collapse_b_to_gene: bool = True,
    label_points: bool = True,
) -> plt.Axes:
    """Compare cooperativity scored at two granularities, e.g. summed versus per-locus ATAC.

    Args:
        coop_a: Cooperativity per gene.
        coop_b: Cooperativity per locus.
        ax: Axis to draw on; created when None.
        labels: Axis labels for the two tables.
        name_col: Gene or locus column.
        value_col: Cooperativity column.
        collapse_b_to_gene: Average coop_b within each gene before joining.
        label_points: Annotate gene names.

    Returns:
        The axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.8, 4.5))

    if coop_a.empty or coop_b.empty:
        return empty_panel(ax, "One table is empty", "Granularity check")

    left = coop_a[[name_col, value_col]].rename(columns={value_col: "_a"})

    right = coop_b[[name_col, value_col]].copy()
    if collapse_b_to_gene:
        right["_gene"] = right[name_col].astype(str).str.split("_").str[0]
        right = right.groupby("_gene", as_index=False)[value_col].mean()
        right = right.rename(columns={"_gene": name_col, value_col: "_b"})
    else:
        right = right.rename(columns={value_col: "_b"})

    merged = left.merge(right, on=name_col).dropna(subset=["_a", "_b"])
    if len(merged) < 3:
        return empty_panel(ax, "Too few shared genes", "Granularity check")

    style_ax(ax)
    x = merged["_a"].to_numpy()
    y = merged["_b"].to_numpy()

    lim = np.array([min(x.min(), y.min()), max(x.max(), y.max())])
    pad = 0.1 * (lim[1] - lim[0] or 1.0)
    lim = lim + np.array([-pad, pad])

    ax.plot(lim, lim, color="gray", lw=1, ls="--", zorder=0)
    ax.axhline(0, color="gray", lw=0.8, ls=":", zorder=0)
    ax.axvline(0, color="gray", lw=0.8, ls=":", zorder=0)

    concordant = np.sign(x) == np.sign(y)
    colors = np.where(concordant,
                      np.where(x < 0, PAL["supra"], PAL["sub"]), PAL["neut"])
    ax.scatter(x, y, c=colors, s=60, edgecolors="white", lw=0.6, zorder=3)

    if label_points:
        place_labels(ax, x, y, merged[name_col])

    rho, p = stats.spearmanr(x, y)
    ax.text(0.04, 0.96,
            f"Spearman $\\rho$ = {rho:.2f}, {pformat(p)}\n"
            f"{int(concordant.sum())}/{len(merged)} agree in sign",
            transform=ax.transAxes, va="top", fontsize=8, color="#444")

    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel(f"$I$, ATAC {labels[0]}", fontsize=9)
    ax.set_ylabel(f"$I$, ATAC {labels[1]}", fontsize=9)
    ax.set_title("Effect of the aggregation choice",
                 fontweight="bold", fontsize=10)
    return ax


# --------------------------------------------------------------------------- #
# Figure assembly
# --------------------------------------------------------------------------- #

def make_main_figure(
    inters: pd.DataFrame,
    coop: pd.DataFrame,
    single_comp: pd.DataFrame,
    pair_comp: pd.DataFrame,
    summaries: dict[str, dict],
    *,
    atac_results: dict[str, dict] | None = None,
    figsize: tuple[float, float] = (16, 13),
) -> plt.Figure:
    """Assemble the six-panel enhancer-level main figure.

    Args:
        inters: Interactions after BH correction.
        coop: build_coop_table output.
        single_comp: Single-enhancer comparison with CRISPRi.
        pair_comp: Enhancer-pair comparison with CRISPRi.
        summaries: summarise_validation output per level.
        atac_results: ATAC GLM results for the interdependency panel.
        figsize: Figure size in inches.

    Returns:
        The figure.
    """
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("#FAFAF8")
    grid = GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.30,
                    left=0.08, right=0.96, top=0.92, bottom=0.06)
    axes = [fig.add_subplot(grid[i, j]) for i in range(3) for j in range(2)]

    plot_additivity(coop, ax=axes[3], readout="RNA")
    plot_interaction_volcano(inters, ax=axes[1])
    plot_gene_interaction_heatmap(inters, ax=axes[2])
    plot_insilico_vs_measured(single_comp, ax=axes[0], level="single")
    plot_insilico_vs_measured(pair_comp, ax=axes[4], level="pair")
    plot_validation_summary(summaries, ax=axes[5])

    panel_letters(axes)
    fig.suptitle("Enhancer cooperativity at K562 target genes",
                 fontsize=14, fontweight="bold", y=0.96)
    return fig


def make_supplementary_figure(
    inters: pd.DataFrame,
    rna_inters: pd.DataFrame,
    atac_inters: pd.DataFrame,
    *,
    comparison: pd.DataFrame | None = None,
    figsize: tuple[float, float] = (15, 9),
) -> plt.Figure:
    """Assemble the five-panel enhancer-level supplementary figure.

    Args:
        inters: Interactions after attach_main_effects and annotate_pair_distance;
            panels without these columns become placeholders.
        rna_inters: Interactions fitted on RNA.
        atac_inters: Interactions fitted on ATAC.
        comparison: build_model_comparison output.
        figsize: Figure size in inches.

    Returns:
        The figure.
    """
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("#FAFAF8")
    # Three rows of two, with the fifth panel occupying the lower left cell.
    grid = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3,
                    left=0.07, right=0.97, top=0.89, bottom=0.09)

    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[2, 0]),
    ]

    if comparison is not None and not comparison.empty:
        plot_model_comparison(comparison, ax=axes[0])
    else:
        empty_panel(axes[0], "No model comparison table", "Model choice")
    plot_saturation_diagnostic(inters, ax=axes[1], against="additive")
    plot_saturation_diagnostic(inters, ax=axes[2], against="baseline")
    plot_interaction_vs_distance(inters, ax=axes[3])
    plot_atac_rna_interaction_concordance(rna_inters, atac_inters, ax=axes[4])

    panel_letters(axes, dx=-0.18)
    fig.suptitle("Supplementary: model choice, artefact checks and mechanism",
                 fontsize=12, fontweight="bold", y=0.955)
    return fig