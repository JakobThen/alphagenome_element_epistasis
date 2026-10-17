"""Figures for the cross-enhancer TF motif analysis.

beta_i is a motif main effect, beta_ij a motif-pair interaction (negative:
super-additive); pair_scope marks whether the two motifs share an enhancer
("within") or not ("across").
"""

from __future__ import annotations

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

SCOPE_COLORS = {"within": PAL["atac"], "across": PAL["crispri"]}
LBL_INTER = "GLM interaction  $\\beta_{ij}$   (log scale)"

# Cells with no estimate. Kept identical to the constant of the same name in
# ``enhancer_plots``: pale blue rather than grey, because grey sits close to the
# midpoint of the diverging colormap and an empty cell would otherwise read as a
# coefficient near zero.
EMPTY_CELL = "#D7E6F2"


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


def _pair_label(pair: str) -> str:
    """Render "GENE_enh1_x_GENE_enh2" as a two-line label: gene over "enh1 x enh2"."""
    sides = str(pair).split("_x_")
    if len(sides) != 2:
        return str(pair).replace("_x_", " x ")

    genes, enhancers = [], []
    for side in sides:
        gene, _, enhancer = side.partition("_")
        genes.append(gene)
        enhancers.append(enhancer or gene)

    # The gene is only redundant when both elements belong to the same one.
    if genes[0] == genes[1] and all(enhancers):
        return f"{genes[0]}\n{enhancers[0]} x {enhancers[1]}"
    return " x ".join(sides)


# --------------------------------------------------------------------------- #
# Effect size against significance
# --------------------------------------------------------------------------- #

def plot_scope_volcanoes(
    inters: pd.DataFrame,
    axes: list[plt.Axes] | None = None,
    value_col: str = "coef",
    qval_col: str = "qval",
    scope_col: str = "pair_scope",
    alpha: float = 0.05,
    label_top: int = 6,
    share_limits: bool = True,
) -> list[plt.Axes]:
    """Volcano plots of within- and across-enhancer motif pairs side by side.

    Args:
        inters: Classified interaction terms.
        axes: Two axes to draw on; created when None.
        value_col: Interaction coefficient column.
        qval_col: Adjusted p-value column.
        scope_col: Pair scope column.
        alpha: Significance threshold.
        label_top: Pairs labelled per direction and panel.
        share_limits: Use the same axis limits in both panels.

    Returns:
        The two axes.
    """
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes = list(np.atleast_1d(axes))

    if inters.empty or qval_col not in inters.columns:
        for ax, scope in zip(axes, ["within", "across"]):
            empty_panel(ax, "No corrected interaction terms", scope)
        return axes

    df = inters.dropna(subset=[value_col, qval_col]).copy()
    df["neglog_q"] = -np.log10(df[qval_col].clip(lower=1e-300))

    if share_limits and not df.empty:
        span = df[value_col].abs().max() * 1.15
        xlim = (-span, span)
        ylim = (-0.4, df["neglog_q"].max() * 1.12 + 0.4)
    else:
        xlim = ylim = None

    for ax, scope in zip(axes, ["within", "across"]):
        block = df[df[scope_col] == scope]
        style_ax(ax)
        if block.empty:
            empty_panel(ax, f"No {scope}-enhancer pairs",
                        f"{scope.capitalize()} enhancer")
            continue

        hit = block[qval_col] < alpha
        colors = np.where(~hit, PAL["neut"],
                          np.where(block[value_col] < 0, PAL["supra"], PAL["sub"]))
        ax.scatter(block[value_col], block["neglog_q"], c=colors, s=42,
                   alpha=0.85, edgecolors="white", lw=0.5, zorder=3)
        ax.axhline(-np.log10(alpha), color="gray", lw=1, ls="--", zorder=0)
        ax.axvline(0, color="gray", lw=0.8, ls=":", zorder=0)

        if label_top and hit.any():
            significant = block[hit]
            chosen = pd.concat([
                significant[significant[value_col] < 0].nsmallest(label_top, value_col),
                significant[significant[value_col] > 0].nlargest(label_top, value_col),
            ])
            if not chosen.empty:
                labels = (chosen["gene"] + "\n" + chosen["motif_a"]
                          + "\u00d7" + chosen["motif_b"])
                place_labels(ax, chosen[value_col], chosen["neglog_q"],
                             labels, fontsize=5.8)

        ax.text(0.03, 0.97,
                f"{int(hit.sum())} of {len(block)} at FDR < {alpha:g}\n"
                f"median $\\beta_{{ij}}$ = {block[value_col].median():+.4f}",
                transform=ax.transAxes, va="top", fontsize=7.5, color="#444")
        add_direction_annotations(ax, supra_corner="lower left",
                                  sub_corner="lower right")

        if xlim:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
        ax.set_xlabel(LBL_INTER, fontsize=9)
        ax.set_ylabel("$-\\log_{10}$ FDR", fontsize=9)
        ax.set_title(f"{scope.capitalize()}-enhancer motif pairs",
                     fontweight="bold", fontsize=10)
    return axes


def plot_scope_comparison(
    inters: pd.DataFrame,
    ax: plt.Axes | None = None,
    value_col: str = "coef",
    scope_col: str = "pair_scope",
    qval_col: str = "qval",
    alpha: float = 0.05,
    use_absolute: bool = False,
) -> plt.Axes:
    """Distribution of interaction coefficients by pair scope, with a Mann-Whitney test.

    Args:
        inters: Classified interaction terms.
        ax: Axis to draw on; created when None.
        value_col: Interaction coefficient column.
        scope_col: Pair scope column.
        qval_col: Adjusted p-value column.
        alpha: Significance threshold.
        use_absolute: Plot absolute instead of signed values.

    Returns:
        The axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 4.4))

    if inters.empty:
        return empty_panel(ax, "No interaction terms", "Within versus across")

    scopes = ["within", "across"]
    data, labels = [], []
    for scope in scopes:
        values = inters.loc[inters[scope_col] == scope,
                            value_col].dropna().to_numpy()
        data.append(np.abs(values) if use_absolute else values)
        labels.append(f"{scope}\n(n={len(values)})")

    if not any(len(d) for d in data):
        return empty_panel(ax, "No values in either scope", "Within versus across")

    style_ax(ax)
    positions = np.arange(len(scopes))
    populated = [i for i, d in enumerate(data) if len(d) > 1]
    if populated:
        parts = ax.violinplot([data[i] for i in populated],
                              positions=positions[populated],
                              showmedians=True, showextrema=False, widths=0.65)
        for body, i in zip(parts["bodies"], populated):
            body.set_facecolor(SCOPE_COLORS[scopes[i]])
            body.set_alpha(0.45)
            body.set_edgecolor("white")
        parts["cmedians"].set_colors("white")
        parts["cmedians"].set_linewidth(2)

    rng = np.random.default_rng(7)
    for i, values in enumerate(data):
        if not len(values):
            continue
        ax.scatter(positions[i] + rng.uniform(-0.08, 0.08, len(values)), values,
                   color=SCOPE_COLORS[scopes[i]], s=16, alpha=0.6,
                   edgecolors="white", lw=0.25, zorder=3)

    if not use_absolute:
        ax.axhline(0, color="gray", lw=1, ls="--", zorder=0)

    lines = []
    if all(len(d) > 2 for d in data):
        pval = stats.mannwhitneyu(data[0], data[1], alternative="two-sided").pvalue
        lines.append(f"Mann-Whitney {pformat(pval)}")
        top = max(d.max() for d in data if len(d))
        rng_span = max(top - min(d.min() for d in data if len(d)), 1e-9)
        ax.plot([0, 0, 1, 1],
                [top + 0.05*rng_span, top + 0.09*rng_span,
                 top + 0.09*rng_span, top + 0.05*rng_span],
                color=PAL["ink"], lw=1)
        ax.text(0.5, top + 0.10*rng_span, stars(pval) or "ns", ha="center",
                va="bottom", fontsize=10, color=PAL["ink"])

    for i, scope in enumerate(scopes):
        block = inters[inters[scope_col] == scope]
        if qval_col in block.columns and len(block):
            lines.append(f"{scope}: {int((block[qval_col] < alpha).sum())} "
                         f"significant")
    ax.text(0.95, 0.8, "\n".join(lines), transform=ax.transAxes, va="top",
            ha = "right", fontsize=7.5, color="#444")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8.5)
    ylabel = ("$|\\beta_{ij}|$" if use_absolute
              else f"$\\beta_{{ij}}$\n{LBL_DIRECTION}")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title("Motif pair interaction by scope", fontweight="bold", fontsize=10)
    return ax


# --------------------------------------------------------------------------- #
# Recurrent grammar
# --------------------------------------------------------------------------- #

def plot_motif_pair_heatmap(
    agg: pd.DataFrame,
    ax: plt.Axes | None = None,
    value_col: str = "mean",
    min_occurrences: int = 1,
    show_range: bool = True,
    vmax: float | None = None,
    title: str = "Motif pair grammar",
    max_factors: int | None = 24,
    aspect: str = "auto",
) -> plt.Axes:
    """Heatmap of mean interaction per factor pair, annotated with count and range.

    Args:
        agg: glm.aggregate_motif_pairs output.
        ax: Axis to draw on; created when None.
        value_col: Column to colour by, normally "mean".
        min_occurrences: Minimum occurrences to show a pair.
        show_range: Print min and max for recurring pairs.
        vmax: Symmetric colour limit, to share a scale between panels.
        title: Panel title.
        max_factors: Keep only the most frequent factors.
        aspect: Cell aspect; "equal" for square cells.

    Returns:
        The axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    if agg is None or agg.empty:
        return empty_panel(ax, "No motif pairs", title)

    work = agg[agg["n"] >= min_occurrences].copy()
    if work.empty:
        return empty_panel(ax, f"No pair seen {min_occurrences}+ times", title)

    # Restrict to the most frequently observed factors, otherwise the matrix is
    # mostly empty and unreadable.
    if max_factors:
        counts = (pd.concat([work.groupby("tf_a")["n"].sum(),
                             work.groupby("tf_b")["n"].sum()])
                    .groupby(level=0).sum().sort_values(ascending=False))
        keep = set(counts.head(max_factors).index)
        work = work[work.tf_a.isin(keep) & work.tf_b.isin(keep)]
        if work.empty:
            return empty_panel(ax, "No pair survives the factor filter", title)

    factors = sorted(set(work["tf_a"]) | set(work["tf_b"]))
    matrix = pd.DataFrame(np.nan, index=factors, columns=factors)
    counts_m = pd.DataFrame(np.nan, index=factors, columns=factors)
    lo_m = pd.DataFrame(np.nan, index=factors, columns=factors)
    hi_m = pd.DataFrame(np.nan, index=factors, columns=factors)

    for _, row in work.iterrows():
        a, b = row["tf_a"], row["tf_b"]
        for i, j in [(a, b), (b, a)]:
            matrix.loc[i, j] = row[value_col]
            counts_m.loc[i, j] = row["n"]
            lo_m.loc[i, j] = row["min"]
            hi_m.loc[i, j] = row["max"]

    style_ax(ax)
    values = matrix.to_numpy(dtype=float)
    limit = vmax if vmax is not None else (
        np.nanmax(np.abs(values)) if np.isfinite(values).any() else 1.0)
    cmap = CMAP_INTERACTION.copy()
    cmap.set_bad(EMPTY_CELL)
    image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap,
                      vmin=-limit, vmax=limit, aspect=aspect)

    n_factors = len(factors)
    font = 5.0 if n_factors > 16 else 6.0
    for i in range(n_factors):
        for j in range(n_factors):
            value = values[i, j]
            if not np.isfinite(value):
                continue
            count = int(counts_m.to_numpy()[i, j])
            text = f"{value:.3f}\n({count})"
            if show_range and count > 1 and n_factors <= 16:
                text += f"\n[{lo_m.to_numpy()[i, j]:.3f},{hi_m.to_numpy()[i, j]:.3f}]"
            ax.text(j, i, text, ha="center", va="center", fontsize=font,
                    color=PAL["ink"])

    ax.set_xticks(range(n_factors))
    ax.set_xticklabels(factors, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n_factors))
    ax.set_yticklabels(factors, fontsize=7)

    bar = plt.colorbar(image, ax=ax, shrink=0.6, pad=0.03)
    bar.set_label(f"Mean $\\beta_{{ij}}$\n{LBL_DIRECTION}", fontsize=8)
    ax.set_title(f"{title}",#\n(cell: mean, occurrences"
    #              f"{', range' if show_range else ''})",
                  fontweight="bold", fontsize=10, y=1.02)
    return ax


def make_grammar_figure(
    inters: pd.DataFrame,
    figsize: tuple[float, float] = (16, 7),
    min_occurrences: int = 1,
    max_factors: int = 20,
    **kwargs: Any,
) -> plt.Figure:
    """Within- and across-enhancer factor-pair heatmaps on a shared colour scale.

    Args:
        inters: Classified interaction terms.
        figsize: Figure size in inches.
        min_occurrences: Minimum occurrences to show a pair.
        max_factors: Factors kept per panel.
        **kwargs: Passed to glm.aggregate_motif_pairs.

    Returns:
        The figure.
    """
    from element_epistasis.glm import aggregate_motif_pairs

    within = aggregate_motif_pairs(inters, scope="within", **kwargs)
    across = aggregate_motif_pairs(inters, scope="across", **kwargs)

    limits = [np.abs(t["mean"]).max() for t in (within, across)
              if t is not None and not t.empty]
    shared = max(limits) if limits else None

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor("#FAFAF8")
    plot_motif_pair_heatmap(within, ax=axes[0], vmax=shared,
                            min_occurrences=min_occurrences,
                            max_factors=max_factors,
                            title="Within enhancer")
    plot_motif_pair_heatmap(across, ax=axes[1], vmax=shared,
                            min_occurrences=min_occurrences,
                            max_factors=max_factors,
                            title="Across enhancers")
    panel_letters(axes, dx=-0.16)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.suptitle("Recurrent motif pair grammar, shared colour scale",
                 fontsize=12, fontweight="bold", y=0.99)
    return fig


# --------------------------------------------------------------------------- #
# Motif to enhancer interdependency
# --------------------------------------------------------------------------- #

def plot_motif_interdependency(
    interdep: pd.DataFrame,
    ax: plt.Axes | None = None,
    genes: list[str] | None = None,
    qval_col: str = "qval",
    alpha: float = 0.05,
    show_only_sig: bool = True,
    hatch_nonsignificant: bool = True,
    annotate: bool = False,
    ylabel_pad: float = 58.0,
    xlabel_pad: float = 10.0,
    gene_label_pad: float = 48.0,
    tick_fontsize: float = 5.5,
) -> plt.Axes:
    """Heatmap of each motif's effect on the accessibility of every enhancer of its gene.

    Rows are motifs grouped by enhancer and gene, columns measured enhancers;
    cells between different genes are undefined and shaded.

    Args:
        interdep: glm.build_motif_interdependency output.
        ax: Axis to draw on; created when None.
        genes: Restrict to these genes.
        qval_col: Adjusted p-value column (falls back to pval).
        alpha: Significance threshold.
        show_only_sig: Show significant cells only and drop empty rows and columns.
        hatch_nonsignificant: Hatch non-significant cells.
        annotate: Print values in cells.
        ylabel_pad: Distance of the y label from the axis in points.
        xlabel_pad: Distance of the x label from the axis in points.
        gene_label_pad: Distance of the gene names from the axis in points.
        tick_fontsize: Font size of the motif labels.

    Returns:
        The axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 12))

    if interdep is None or interdep.empty:
        return empty_panel(ax, "No motif interdependency table",
                           "Motif interdependency")

    work = interdep.copy()
    # The table now carries both levels; this panel shows single motifs only.
    if "level" in work.columns:
        work = work[work["level"] == "main"]
    if genes is not None:
        work = work[work["gene"].isin(genes)]
    if work.empty:
        return empty_panel(ax, "No rows for the selected genes",
                           "Motif interdependency")

    # Rows ordered by gene, then by parent enhancer, so cis cells form blocks.
    # gene_rows and gene_cols map an axis label to its gene, so the block
    # boundaries survive the filtering applied below.
    row_order, col_order, gene_rows, gene_cols = [], [], {}, {}
    for gene, block in work.groupby("gene", sort=True):
        # Order rows by parent enhancer so cis cells form visible blocks.
        # Derived here rather than required as a column, since the two-level
        # table cannot carry a single parent enhancer for pair rows.
        block = block.copy()
        block["_parent"] = block["gene"] + "_" + block["enh_a"].astype(str)
        motifs = (block[["perturbed", "_parent"]].drop_duplicates()
                       .sort_values(["_parent", "perturbed"])["perturbed"]
                       .tolist())
        loci = sorted(block["measured"].unique())
        row_order.extend(motifs)
        col_order.extend(loci)
        gene_rows.update({name: gene for name in motifs})
        gene_cols.update({name: gene for name in loci})

    row_idx = {name: i for i, name in enumerate(row_order)}
    col_idx = {name: i for i, name in enumerate(col_order)}
    matrix = np.full((len(row_order), len(col_order)), np.nan)
    qmatrix = np.full_like(matrix, np.nan)

    qcol = qval_col if qval_col in work.columns else "pval"
    for _, row in work.iterrows():
        i, j = row_idx.get(row["perturbed"]), col_idx.get(row["measured"])
        if i is None or j is None:
            continue
        matrix[i, j] = row["delta"]
        qmatrix[i, j] = row.get(qcol, np.nan)

    # Filter before the colour scale is fixed, so a discarded cell cannot set
    # the limit and flatten the contrast of everything that survives.
    if show_only_sig:
        passes = np.isfinite(qmatrix) & (qmatrix < alpha)
        matrix = np.where(passes, matrix, np.nan)
        qmatrix = np.where(passes, qmatrix, np.nan)

        keep_rows = np.isfinite(matrix).any(axis=1)
        keep_cols = np.isfinite(matrix).any(axis=0)
        if not keep_rows.any():
            return empty_panel(ax, f"No cell at FDR < {alpha:g}",
                               "Motif interdependency")

        matrix = matrix[np.ix_(keep_rows, keep_cols)]
        qmatrix = qmatrix[np.ix_(keep_rows, keep_cols)]
        row_order = [r for r, keep in zip(row_order, keep_rows) if keep]
        col_order = [c for c, keep in zip(col_order, keep_cols) if keep]

    style_ax(ax)
    limit = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1.0
    cmap = CMAP_INTERACTION.copy()
    cmap.set_bad(EMPTY_CELL)

    image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap,
                      vmin=-limit, vmax=limit, aspect="auto")

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

    if annotate:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center",
                            va="center", fontsize=5, color=PAL["ink"])

    # Strip the gene prefix; the gene is shown once per block instead.
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels([c.split("_", 1)[-1] for c in col_order],
                       rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels([r.split("_", 1)[-1] for r in row_order],
                       fontsize=tick_fontsize)
    ax.set_xlabel("Measured enhancer", fontsize=9, fontweight="bold",
                  labelpad=xlabel_pad)
    ax.set_ylabel("Perturbed motif", fontsize=9, fontweight="bold",
                  labelpad=ylabel_pad)

    # Rows and columns are filtered independently, so the two sets of
    # separators are derived from their own label lists.
    cursor = 0
    for gene, count in _gene_blocks(row_order, gene_rows):
        if cursor:
            ax.axhline(cursor - 0.5, color="white", lw=1.4, zorder=5)
        # Offset in points, so the gene names clear the motif labels at any
        # figure size rather than only at the one this was tuned on.
        ax.annotate(gene, xy=(0, cursor + count / 2 - 0.5),
                    xycoords=ax.get_yaxis_transform(),
                    xytext=(-gene_label_pad, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=6.5,
                    fontweight="bold", color=PAL["ink"])
        cursor += count

    cursor = 0
    for _, count in _gene_blocks(col_order, gene_cols):
        if cursor:
            ax.axvline(cursor - 0.5, color="white", lw=1.4, zorder=5)
        cursor += count

    bar = plt.colorbar(image, ax=ax, shrink=0.35, pad=0.02)
    bar.set_label("$\\beta_i$, ATAC (log scale)", fontsize=8)
    if show_only_sig:
        note = f"FDR < {alpha:g} only"
    elif hatch_nonsignificant:
        note = f"hatched: FDR $\\geq$ {alpha:g}"
    else:
        note = "all estimated cells"
    ax.set_title(f"Motif influence on enhancer accessibility ({note})",
                 fontweight="bold", fontsize=10, pad=10)
    return ax


def plot_pair_interdependency(
    interdep: pd.DataFrame,
    ax: plt.Axes | None = None,
    genes: list[str] | None = None,
    qval_col: str = "qval",
    alpha: float = 0.05,
    significant_only: bool = True,
    max_rows: int = 60,
    annotate: bool = False,
) -> plt.Axes:
    """Heatmap of motif-pair interactions on each enhancer's accessibility; row colour marks cis, mixed or trans.

    Args:
        interdep: glm.build_motif_interdependency output.
        ax: Axis to draw on; created when None.
        genes: Restrict to these genes.
        qval_col: Adjusted p-value column (falls back to pval).
        alpha: Significance threshold.
        significant_only: Keep pairs with at least one significant cell.
        max_rows: Maximum pairs shown, by absolute effect.
        annotate: Print values in cells.

    Returns:
        The axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 10))

    if interdep is None or interdep.empty or "level" not in interdep.columns:
        return empty_panel(ax, "No pairwise interdependency terms",
                           "Motif pair interdependency")

    work = interdep[interdep["level"] == "interaction"].copy()
    if genes is not None:
        work = work[work["gene"].isin(genes)]
    if work.empty:
        return empty_panel(ax, "No pairwise terms for the selection",
                           "Motif pair interdependency")

    qcol = qval_col if qval_col in work.columns else "pval"
    if significant_only and qcol in work.columns:
        keep_pairs = set(work.loc[work[qcol] < alpha, "perturbed"])
        work = work[work["perturbed"].isin(keep_pairs)]
        if work.empty:
            return empty_panel(ax, f"No pair reaches FDR < {alpha:g}",
                               "Motif pair interdependency")

    if max_rows:
        strength = (work.groupby("perturbed")["delta"].apply(
            lambda v: v.abs().max()).sort_values(ascending=False))
        work = work[work["perturbed"].isin(strength.head(max_rows).index)]

    row_order, col_order = [], []
    for gene, block in work.groupby("gene", sort=True):
        row_order.extend(sorted(block["perturbed"].unique()))
        col_order.extend(sorted(block["measured"].unique()))
    row_idx = {name: i for i, name in enumerate(row_order)}
    col_idx = {name: i for i, name in enumerate(col_order)}

    matrix = np.full((len(row_order), len(col_order)), np.nan)
    qmatrix = np.full_like(matrix, np.nan)
    classes = np.empty((len(row_order), len(col_order)), dtype=object)
    for _, row in work.iterrows():
        i, j = row_idx.get(row["perturbed"]), col_idx.get(row["measured"])
        if i is None or j is None:
            continue
        matrix[i, j] = row["delta"]
        qmatrix[i, j] = row.get(qcol, np.nan)
        classes[i, j] = row["cis_class"]

    style_ax(ax)
    limit = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1.0
    cmap = CMAP_INTERACTION.copy()
    cmap.set_bad("#E8E8E6")
    alphas = np.ones_like(matrix)
    alphas[np.isfinite(qmatrix) & (qmatrix >= alpha)] = 0.25

    image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap,
                      vmin=-limit, vmax=limit, aspect="auto",
                      alpha=np.ma.masked_invalid(alphas).filled(1.0))

    # Mark which cells are cis, mixed or trans with a small corner glyph, since
    # the distinction is the point of the panel and cannot be read off colour.
    glyph = {"cis": "o", "mixed": "+", "trans": "."}
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if classes[i, j] in glyph:
                ax.text(j - 0.34, i - 0.30, glyph[classes[i, j]], fontsize=5,
                        color="#444", ha="center", va="center")
            if annotate and np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        fontsize=5, color=PAL["ink"])

    def prettify(pair: str) -> str:
        return " x ".join(part.split("_", 1)[-1] for part in pair.split(" x "))

    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels([c.split("_", 1)[-1] for c in col_order],
                       rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels([prettify(r) for r in row_order], fontsize=5)
    ax.set_xlabel("Measured enhancer", fontsize=9)
    ax.set_ylabel("Perturbed motif pair", fontsize=9)

    ax.legend(handles=[
        mpatches.Patch(color="none", label="o  both motifs in measured enhancer"),
        mpatches.Patch(color="none", label="+  one motif elsewhere"),
        mpatches.Patch(color="none", label=".  both motifs elsewhere"),
    ], fontsize=6, loc="lower left", framealpha=0.9, handlelength=0)

    bar = plt.colorbar(image, ax=ax, shrink=0.35, pad=0.02)
    bar.set_label(f"$\\beta_{{ij}}$, ATAC\n{LBL_DIRECTION}", fontsize=8)
    ax.set_title(f"Motif pair influence on enhancer accessibility "
                 f"(faded: FDR $\\geq$ {alpha:g})",
                 fontweight="bold", fontsize=10, pad=10)
    return ax


def plot_cis_trans_summary(
    interdep: pd.DataFrame,
    ax: plt.Axes | None = None,
    qval_col: str = "qval",
    alpha: float = 0.05,
    include_interactions: bool = True,
) -> plt.Axes:
    """Violins of motif effects on accessibility by level (main, pair) and class (cis, mixed, trans).

    Args:
        interdep: glm.build_motif_interdependency output.
        ax: Axis to draw on; created when None.
        qval_col: Adjusted p-value column.
        alpha: Significance threshold.
        include_interactions: Add the motif-pair groups.

    Returns:
        The axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 4.4))

    if interdep is None or interdep.empty:
        return empty_panel(ax, "No interdependency table", "Cis versus trans")

    has_level = "level" in interdep.columns
    wanted = [("main", "cis", "main\ncis", PAL["atac"]),
              ("main", "trans", "main\ntrans", PAL["crispri"])]
    if include_interactions and has_level:
        wanted += [("interaction", "cis", "pair\ncis", PAL["supra"]),
                   ("interaction", "mixed", "pair\nmixed", PAL["ink"]),
                   ("interaction", "trans", "pair\ntrans", PAL["sub"])]

    groups, labels, colors, keys = [], [], [], []
    for level, cis_class, label, color in wanted:
        block = interdep
        if has_level:
            block = block[block["level"] == level]
        if "cis_class" in block.columns:
            block = block[block["cis_class"] == cis_class]
        elif cis_class in ("cis", "trans"):
            block = block[block["is_cis"] == (cis_class == "cis")]
        values = block["delta"].dropna().to_numpy()
        if not len(values):
            continue
        groups.append(values)
        labels.append(f"{label}\n(n={len(values)})")
        colors.append(color)
        keys.append((level, cis_class, block))

    if not groups:
        return empty_panel(ax, "No values", "Cis versus trans")

    style_ax(ax)
    positions = np.arange(len(groups))
    populated = [i for i, g in enumerate(groups) if len(g) > 1]
    if populated:
        parts = ax.violinplot([groups[i] for i in populated],
                              positions=positions[populated],
                              showmedians=True, showextrema=False, widths=0.65)
        for body, i in zip(parts["bodies"], populated):
            body.set_facecolor(colors[i])
            body.set_alpha(0.45)
            body.set_edgecolor("white")
        parts["cmedians"].set_colors("white")
        parts["cmedians"].set_linewidth(2)

    rng = np.random.default_rng(5)
    for i, values in enumerate(groups):
        ax.scatter(positions[i] + rng.uniform(-0.08, 0.08, len(values)), values,
                   color=colors[i], s=12, alpha=0.5, edgecolors="white",
                   lw=0.15, zorder=3)
    ax.axhline(0, color="gray", lw=1, ls="--", zorder=0)

    # Significance count per group, printed above each violin.
    for i, (level, cis_class, block) in enumerate(keys):
        if qval_col in block.columns and len(block):
            n_sig = int((block[qval_col] < alpha).sum())
            ax.text(positions[i], ax.get_ylim()[1],
                    f"{n_sig}/{len(block)}", ha="center", va="top",
                    fontsize=6, color="#555")

    lines = []
    if len(groups) >= 2 and all(len(g) for g in groups[:2]):
        ratio = abs(np.median(groups[0])) / max(abs(np.median(groups[1])), 1e-12)
        lines.append(f"main: cis {np.median(groups[0]):+.3f}  "
                     f"trans {np.median(groups[1]):+.3f}  ({ratio:.0f}x)")
    if len(groups) >= 5:
        lines.append(f"pair: cis {np.median(groups[2]):+.4f}  "
                     f"mixed {np.median(groups[3]):+.4f}  "
                     f"trans {np.median(groups[4]):+.4f}")
    if lines:
        ax.text(0.02, 0.02, "\n".join(lines), transform=ax.transAxes,
                va="bottom", fontsize=7, color="#444", family="monospace")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("Coefficient, ATAC (log scale)", fontsize=9)
    ax.set_title("Motif effects by level and spatial relation",
                 fontweight="bold", fontsize=10)
    return ax


def make_interdependency_figure(
    interdep: pd.DataFrame,
    *,
    genes: list[str] | None = None,
    figsize: tuple[float, float] = (17, 10),
    **kwargs: Any,
) -> plt.Figure:
    """Assemble the single-motif and motif-pair interdependency heatmaps and the summary panel.

    Args:
        interdep: glm.build_motif_interdependency output.
        genes: Restrict the heatmaps to these genes.
        figsize: Figure size in inches.
        **kwargs: Passed to plot_pair_interdependency.

    Returns:
        The figure.
    """
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("#FAFAF8")
    grid = GridSpec(1, 3, figure=fig, width_ratios=[1.15, 1.15, 0.85],
                    wspace=0.42, left=0.07, right=0.97, top=0.90, bottom=0.10)

    ax_main = fig.add_subplot(grid[0, 0])
    ax_pair = fig.add_subplot(grid[0, 1])
    ax_sum = fig.add_subplot(grid[0, 2])

    plot_motif_interdependency(interdep, ax=ax_main, genes=genes)
    plot_pair_interdependency(interdep, ax=ax_pair, genes=genes, **kwargs)
    plot_cis_trans_summary(interdep, ax=ax_sum)

    panel_letters([ax_main, ax_pair, ax_sum], dx=-0.2)
    fig.suptitle("Motif influence on enhancer accessibility, "
                 "singly and in pairs",
                 fontsize=12, fontweight="bold", y=0.965)
    return fig


# --------------------------------------------------------------------------- #
# Mechanism and cross-level
# --------------------------------------------------------------------------- #

def plot_interaction_vs_motif_distance(
    inters: pd.DataFrame,
    ax: plt.Axes | None = None,
    value_col: str = "coef",
    distance_col: str = "distance_bp",
    scope_col: str = "pair_scope",
    qval_col: str = "qval",
    alpha: float = 0.05,
) -> plt.Axes:
    """Interaction against the distance between motif midpoints, coloured by pair scope.

    Args:
        inters: Classified interaction terms with distance_bp.
        ax: Axis to draw on; created when None.
        value_col: Interaction coefficient column.
        distance_col: Distance column (bp).
        scope_col: Pair scope column.
        qval_col: Adjusted p-value column.
        alpha: Significance threshold.

    Returns:
        The axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.2))

    if inters.empty or distance_col not in inters.columns:
        return empty_panel(ax, "No motif distances attached",
                           "Interaction versus distance")

    df = inters.dropna(subset=[distance_col, value_col])
    df = df[df[distance_col] > 0]
    if len(df) < 3:
        return empty_panel(ax, "Too few pairs", "Interaction versus distance")

    style_ax(ax)
    for scope in ["within", "across"]:
        block = df[df[scope_col] == scope]
        if block.empty:
            continue
        ax.scatter(block[distance_col], block[value_col],
                   color=SCOPE_COLORS[scope], s=32, alpha=0.7,
                   edgecolors="white", lw=0.4, zorder=3, label=scope)
    ax.axhline(0, color="gray", lw=0.8, ls="--", zorder=0)
    ax.set_xscale("log")

    rho, pval = stats.spearmanr(df[distance_col], df[value_col])
    ax.text(0.95, 0.8,
            f"all pairs: Spearman $\\rho$ = {rho:.3f}  {pformat(pval)}\n"
            f"n = {len(df)}",
            transform=ax.transAxes, va="top", ha = "right", fontsize=7.5, color="#444")

    ax.legend(fontsize=7, loc="upper right", framealpha=0.85)
    ax.set_xlabel("Distance between motif midpoints (bp)", fontsize=9)
    ax.set_ylabel(LBL_INTER, fontsize=9)
    ax.set_title("Distance dependence across scales",
                 fontweight="bold", fontsize=10)
    return ax


def plot_motif_vs_enhancer_interaction(
    pair_agg: pd.DataFrame,
    enh_inters: pd.DataFrame,
    ax: plt.Axes | None = None,
    motif_col: str = "motif_sum",
    enh_value_col: str = "coef",
    label_points: bool = True,
    label_top: int = 10,
    title: str | None = None,
) -> plt.Axes:
    """Aggregated across-enhancer motif interaction against the enhancer-level interaction.

    Args:
        pair_agg: glm.aggregate_pairs_to_enhancer output.
        enh_inters: Enhancer-level interactions with enhA and enhB (in silico or measured).
        ax: Axis to draw on; created when None.
        motif_col: Motif-side aggregate column.
        enh_value_col: Enhancer-level coefficient column.
        label_points: Annotate enhancer pairs.
        label_top: Number of pairs labelled, ranked by distance from the origin.
        title: Panel title.

    Returns:
        The axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.8, 4.4))

    if pair_agg is None or pair_agg.empty or enh_inters is None or enh_inters.empty:
        return empty_panel(ax, "One level is missing", "Motif versus enhancer")

    if title is None:
        title = "Does motif grammar explain enhancer interaction?"

    def key(frame: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
        pairs = frame.apply(
            lambda r: tuple(sorted([str(r[a]), str(r[b])])), axis=1)
        return pd.DataFrame({"_a": [p[0] for p in pairs],
                             "_b": [p[1] for p in pairs]}, index=frame.index)

    left = pair_agg.join(key(pair_agg, "enhA", "enhB"))
    right = enh_inters.join(key(enh_inters, "full_a", "full_b")) \
        if {"full_a", "full_b"}.issubset(enh_inters.columns) \
        else enh_inters.join(key(enh_inters, "enhA", "enhB"))

    merged = left.merge(right[["_a", "_b", enh_value_col]], on=["_a", "_b"],
                        how="inner").dropna(subset=[motif_col, enh_value_col])
    if len(merged) < 3:
        return empty_panel(ax, f"Only {len(merged)} matched pair(s)",
                           "Motif versus enhancer")

    style_ax(ax)
    x = merged[motif_col].to_numpy()
    y = merged[enh_value_col].to_numpy()

    ax.axhline(0, color="gray", lw=0.8, ls="--", zorder=0)
    ax.axvline(0, color="gray", lw=0.8, ls="--", zorder=0)
    colors = [PAL["supra"] if v < 0 else PAL["sub"] for v in y]
    ax.scatter(x, y, c=colors, s=55, alpha=0.85, edgecolors="white",
               lw=0.6, zorder=3)

    if label_points and label_top and "enh_pair" in merged.columns:
        names = merged["enh_pair"].map(_pair_label).to_numpy()
        chosen = np.argsort(-np.hypot(x, y))[:label_top]
        place_labels(ax, x[chosen], y[chosen], names[chosen], fontsize=5.5)

    if len(np.unique(x)) > 1 and len(np.unique(y)) > 1:
        slope, intercept, r, p, _ = stats.linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, intercept + slope * xs, color=PAL["ink"], lw=1.2, zorder=2)
        rho, p_rho = stats.spearmanr(x, y)
        concordant = float((np.sign(x) == np.sign(y)).mean())
        ax.text(0.03, 0.95,
                f"r = {r:+.3f}  {pformat(p)}\n"
                f"rho = {rho:+.3f}  {pformat(p_rho)}\n"
                f"{concordant:.0%} sign concordant, n = {len(merged)}",
                transform=ax.transAxes, va="top", fontsize=7.5,
                color="#444", family="monospace")

    pretty = motif_col.replace("motif_", "").replace("_", " ")
    ax.set_xlabel(f"Cross-enhancer motif interaction ({pretty})", fontsize=9)
    ax.set_ylabel("Enhancer-level  $\\beta_{ij}$", fontsize=9)
    ax.set_title(f"{title}", fontweight="bold", fontsize=10)
    return ax


# --------------------------------------------------------------------------- #
# Figure assembly
# --------------------------------------------------------------------------- #

def _scope_volcano(
    inters: pd.DataFrame,
    scope: str,
    ax: plt.Axes,
    scope_col: str = "pair_scope",
    **kwargs: Any,
) -> plt.Axes:
    """Draw one pair scope with the enhancer-level volcano function, using motif-pair labels."""
    from element_epistasis import enhancer_plots as ep

    block = inters[inters[scope_col] == scope].copy() if scope_col in inters \
        else inters.copy()
    if block.empty:
        return empty_panel(ax, f"No {scope}-enhancer pairs",
                           f"{scope.capitalize()}-enhancer motif pairs")

    if {"gene", "motif_a", "motif_b"}.issubset(block.columns):
        block["_pair_label"] = (block["gene"].astype(str) + "\n"
                                + block["motif_a"].astype(str) + "\u00d7"
                                + block["motif_b"].astype(str))
        kwargs.setdefault("label_col", "_pair_label")

    # Ten labels per direction is legible on the enhancer-level volcano, which
    # has far fewer points than a motif pair table; five keeps these readable.
    kwargs.setdefault("label_top", 5)

    ep.plot_interaction_volcano(block, ax=ax, **kwargs)
    ax.set_title(f"{scope.capitalize()}-enhancer motif pairs",
                 fontweight="bold", fontsize=10)
    return ax


def make_main_figure(
    inters_rna: pd.DataFrame,
    interdep: pd.DataFrame,
    *,
    pair_agg: pd.DataFrame | None = None,
    enh_inters: pd.DataFrame | None = None,
    measured_pairs: pd.DataFrame | None = None,
    motif_col: str = "motif_mean",
    min_occurrences: int = 2,
    max_factors: int = 50,
    figsize: tuple[float, float] = (10, 14),
    suptitle: str | None = None,
) -> plt.Figure:
    """Assemble the seven-panel cross-enhancer main figure.

    Panels with missing inputs are drawn as placeholders.

    Args:
        inters_rna: Classified interaction terms on expression.
        interdep: Motif interdependency table on accessibility.
        pair_agg: Enhancer-pair aggregation of motif interactions.
        enh_inters: In silico enhancer-level interactions.
        measured_pairs: Measured interactions, coefficient in a "coef" column.
        motif_col: Motif-side aggregate for the cross-level panels.
        min_occurrences: Minimum occurrences for the grammar heatmap.
        max_factors: Factors kept in the grammar heatmap.
        figsize: Figure size in inches.
        suptitle: Figure title; none by default.

    Returns:
        The figure.
    """
    from element_epistasis.glm import aggregate_motif_pairs

    # Pairs never seen to interact carry no grammar and only dilute the matrix.
    within = aggregate_motif_pairs(inters_rna, scope="within")
    if within is not None and not within.empty and "n_significant" in within:
        within = within[within["n_significant"] > 0]

    fig, axes = plt.subplots(4, 2, figsize=figsize)
    fig.patch.set_facecolor("#FAFAF8")

    plot_scope_comparison(inters_rna, ax=axes[0, 0])
    _scope_volcano(inters_rna, "across", axes[0, 1])
    plot_interaction_vs_motif_distance(inters_rna, ax=axes[1, 0])
    _scope_volcano(inters_rna, "within", axes[1, 1])

    if within is not None and not within.empty:
        plot_motif_pair_heatmap(within, ax=axes[2, 0],
                                vmax=float(within["mean"].abs().max()),
                                min_occurrences=min_occurrences,
                                max_factors=max_factors,
                                title="Within enhancer")
    else:
        empty_panel(axes[2, 0], "No recurrent within-enhancer pairs",
                    "Within enhancer")

    plot_cis_trans_summary(interdep, ax=axes[2, 1])

    cross_level = [
        (axes[3, 0], enh_inters, "Motif vs. Enhancer Interaction in silico"),
        (axes[3, 1], measured_pairs, "Motif vs. Enhancer Interaction CRISPRi"),
    ]
    for ax, table, title in cross_level:
        if pair_agg is not None and table is not None and not table.empty:
            plot_motif_vs_enhancer_interaction(pair_agg, table, ax=ax,
                                               motif_col=motif_col, title=title)
        else:
            empty_panel(ax, "Cross-level inputs not supplied", title)

    # The two volcanoes are read as one panel, so the letter on the second is
    # skipped and the sequence resumes on the third row.
    lettered = [(axes[0, 0], "A"), (axes[0, 1], "B"), (axes[1, 0], "C"),
                (axes[2, 0], "D"), (axes[2, 1], "E"),
                (axes[3, 0], "F"), (axes[3, 1], "G")]
    for ax, letter in lettered:
        ax.text(-0.16, 1.06, letter, transform=ax.transAxes, fontsize=14,
                fontweight="bold", va="bottom")

    top = 0.955 if suptitle else 1.0
    fig.tight_layout(rect=(0, 0, 1, top))
    # tight_layout packs the panels tightly; the annotation blocks and the
    # rotated factor labels need more room than it leaves.
    fig.subplots_adjust(hspace=0.55, wspace=0.32)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.985)
    return fig