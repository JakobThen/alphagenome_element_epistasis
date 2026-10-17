"""Figures for the within-enhancer TF motif analysis.

Each panel function takes a tf_cooperativity table and an optional axis and
returns the axis. Negative interaction or cooperativity is super-additive.
"""

from __future__ import annotations

import itertools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy import stats

from element_epistasis.plot_style import (
    CMAP_INTERACTION, LBL_DELTA, LBL_DIRECTION, LBL_I, PAL,
    add_direction_annotations, empty_panel, panel_letters,
    pformat, stars, style_ax,
)
from element_epistasis.plot_style import label_points as place_labels
from element_epistasis.tf_cooperativity import partial_correlation


def plot_effect_vs_attribution(
    motifs: pd.DataFrame,
    ax: plt.Axes | None = None,
    readout: str = "ATAC",
) -> plt.Axes:
    """Scatter of single-motif scramble effect against ISM attribution.

    Args:
        motifs: build_motif_effects output with attribution attached.
        ax: Axis to draw on; created when None.
        readout: Readout name for the axis label.

    Returns:
        The axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))

    if motifs.empty or "attribution" not in motifs.columns:
        return empty_panel(ax, "No attribution column", "Seqlet validation")

    df = motifs.dropna(subset=["attribution", "delta"])
    if len(df) < 3:
        return empty_panel(ax, "Too few motifs", "Seqlet validation")

    style_ax(ax)
    ax.scatter(df["attribution"], df["delta"], s=45, color=PAL["atac"],
               alpha=0.75, edgecolors="white", lw=0.5, zorder=3)
    ax.axhline(0, color="gray", lw=0.8, ls="--", zorder=0)

    slope, intercept, r, p, _ = stats.linregress(df["attribution"], df["delta"])
    xs = np.linspace(df["attribution"].min(), df["attribution"].max(), 50)
    ax.plot(xs, intercept + slope * xs, color=PAL["ink"], lw=1.2, zorder=2)

    pstr = f"p = {p:.1e}" if p < 0.001 else f"p = {p:.3f}"
    ax.text(0.04, 0.06, f"r = {r:.2f},  {pstr}\nn = {len(df)} motifs",
            transform=ax.transAxes, va="bottom", fontsize=8, color="#444")

    ax.set_xlabel("Seqlet ISM attribution", fontsize=9)
    ax.set_ylabel(f"$\\delta_i = \\log(y_i / y_0)$, {readout}", fontsize=9)
    ax.set_title("Attribution predicts scramble effect",
                 fontweight="bold", fontsize=10)
    return ax


def plot_motif_importance(
    motifs: pd.DataFrame,
    ax: plt.Axes | None = None,
    qval_col: str = "qval",
    min_occurrences: int = 1,
    top_n: int | None = None,
    controls: pd.DataFrame | None = None,
) -> plt.Axes:
    """Bar chart of the mean single-motif effect per transcription factor.

    Labels with a "#" suffix are pooled into their parent factor.

    Args:
        motifs: build_motif_effects output.
        ax: Axis to draw on; created when None.
        qval_col: Adjusted p-value column for significance markers.
        min_occurrences: Minimum number of enhancers per factor.
        top_n: Keep the strongest factors by absolute mean effect.
        controls: Control-window effects; their IQR is shaded.

    Returns:
        The axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 6))

    if motifs.empty:
        return empty_panel(ax, "No motif effects", "Per-motif importance")

    df = motifs.copy()
    df["tf"] = df["motif"].str.split("#").str[0]

    agg = (
        df.groupby("tf")
        .agg(mean=("delta", "mean"), sem=("delta", "sem"), n=("delta", "size"))
        .reset_index()
    )
    agg = agg[agg["n"] >= min_occurrences]
    if agg.empty:
        return empty_panel(ax, "No factor passes the occurrence filter",
                      "Per-motif importance")

    if top_n:
        agg = agg.reindex(agg["mean"].abs().sort_values(ascending=False).index)
        agg = agg.head(top_n)
    agg = agg.sort_values("mean", ascending=False)

    style_ax(ax)
    y = np.arange(len(agg))
    colors = [PAL["atac"] if v < 0 else PAL["neut"] for v in agg["mean"]]

    ax.barh(y, agg["mean"], color=colors, height=0.65,
            edgecolor="white", lw=0.5,
            xerr=agg["sem"].fillna(0),
            error_kw=dict(lw=1, capsize=3, color="#555"))
    ax.axvline(0, color="gray", lw=0.8, zorder=0)

    if controls is not None and not controls.empty:
        lo, hi = np.percentile(controls["delta"].dropna(), [25, 75])
        ax.axvspan(lo, hi, color=PAL["neut"], alpha=0.28, zorder=0)
        ax.text(0.5 * (lo + hi), len(agg) - 0.3, "control IQR", ha="center",
                va="bottom", fontsize=6.5, color="#666")

    ax.set_yticks(y)
    ax.set_yticklabels(agg["tf"], fontsize=8.5)

    span = agg["mean"].abs().max()
    for i, (value, n) in enumerate(zip(agg["mean"], agg["n"])):
        offset = span * 0.03
        ax.text(value - offset if value < 0 else value + offset, i, f"n={n}",
                va="center", ha="right" if value < 0 else "left",
                fontsize=6.5, color="#777")

    ax.set_xlabel("Mean $\\delta_i = \\log(y_i / y_0)$, ATAC ($\\pm$ SEM)",
                  fontsize=9)
    ax.set_title("Per-motif importance", fontweight="bold", fontsize=10)
    return ax


def plot_additivity(
    coop: pd.DataFrame,
    ax: plt.Axes | None = None,
    readout: str = "ATAC",
    label_points: bool = True,
    label_top: int = 10,
) -> plt.Axes:
    """Observed all-scrambled effect against the additive prediction, one point per enhancer.

    Args:
        coop: build_cooperativity_table output.
        ax: Axis to draw on; created when None.
        readout: Readout name for the axis labels.
        label_points: Annotate enhancer names.
        label_top: Number of enhancers labelled, ranked by orthogonal distance
            from the diagonal.

    Returns:
        The axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    df = coop[coop["n_seqlets"] >= 2].dropna(subset=["additive_prediction", "observed"])
    if df.empty:
        return empty_panel(ax, "No enhancer with two or more motifs",
                      f"Additivity ({readout})")

    style_ax(ax)
    x = df["additive_prediction"].to_numpy()
    y = df["observed"].to_numpy()

    lim = np.array([min(x.min(), y.min()), max(x.max(), y.max())])
    pad = 0.08 * (lim[1] - lim[0] or 1.0)
    lim = lim + np.array([-pad, pad])

    ax.plot(lim, lim, color="gray", lw=1, ls="--", zorder=0)
    ax.fill_between(lim, lim, lim[1], color=PAL["sub"], alpha=0.05, zorder=0)
    ax.fill_between(lim, lim[0], lim, color=PAL["supra"], alpha=0.05, zorder=0)

    colors = np.where(y > x, PAL["sub"], PAL["supra"])
    ax.scatter(x, y, c=colors, s=65, edgecolors="white", lw=0.6, zorder=3)

    if label_points and label_top:
        # Orthogonal distance to y = x is |y - x| / sqrt(2); the constant factor
        # does not change the ranking, so the absolute residual is enough.
        names = df["enhancer"].to_numpy()
        chosen = np.argsort(-np.abs(y - x))[:label_top]
        place_labels(ax, x[chosen], y[chosen], names[chosen])

    r, p = stats.pearsonr(x, y)
    pstr = f"p = {p:.1e}" if p < 0.001 else f"p = {p:.3f}"
    ax.text(0.04, 0.95, f"r = {r:.2f},  {pstr}\nn = {len(df)} enhancers",
            transform=ax.transAxes, va="top", fontsize=8, color="#444")

    ax.text(0.96, 0.05, "super-additive", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=9, color=PAL["supra"], style="italic")
    ax.text(0.05, 0.72, "sub-additive", transform=ax.transAxes, ha="left",
            va="top", fontsize=9, color=PAL["sub"], style="italic")

    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel(f"Additive prediction  $\\Sigma_i\\, \\delta_i$   ({readout})", fontsize=9)
    ax.set_ylabel(f"Observed  $\\delta_{{\\mathrm{{all}}}}$   ({readout})", fontsize=9)
    ax.set_title(f"Deviation from independence ({readout})",
                 fontweight="bold", fontsize=10)
    return ax

def plot_cooperativity_forest(
    coop: pd.DataFrame,
    ax: plt.Axes | None = None,
    qval_col: str = "qval",
    top_n: int | None = 25,
) -> plt.Axes:
    """Forest plot of per-enhancer cooperativity with confidence intervals.

    Args:
        coop: build_cooperativity_table output, ideally BH corrected.
        ax: Axis to draw on; created when None.
        qval_col: Adjusted p-value column.
        top_n: Show the strongest enhancers by absolute score.

    Returns:
        The axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 7))

    df = coop.dropna(subset=["cooperativity"])
    if df.empty:
        return empty_panel(ax, "No cooperativity scores", "Cooperativity per enhancer")

    if top_n and len(df) > top_n:
        df = df.reindex(df["cooperativity"].abs().sort_values(ascending=False).index)
        df = df.head(top_n)
    df = df.sort_values("cooperativity")

    style_ax(ax)
    y = np.arange(len(df))
    err = 1.96 * df["sem"].fillna(0)
    colors = [PAL["supra"] if v < 0 else PAL["sub"] for v in df["cooperativity"]]

    ax.errorbar(df["cooperativity"], y, xerr=err, fmt="none",
                ecolor="#888", elinewidth=1, capsize=2, zorder=2)
    ax.scatter(df["cooperativity"], y, c=colors, s=45,
               edgecolors="white", lw=0.6, zorder=3)
    ax.axvline(0, color="gray", lw=1, ls="--", zorder=0)

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{e}  ({int(n)})" for e, n in zip(df["enhancer"], df["n_seqlets"])],
        fontsize=7.5,
    )

    if qval_col in df.columns:
        span = (df["cooperativity"].abs() + err).max()
        for i, q in enumerate(df[qval_col]):
            marker = stars(q)
            if marker:
                ax.text(span * 1.05, i, marker, va="center", ha="left",
                        fontsize=9, color=PAL["ink"])

    ax.set_xlabel("$I = \\delta_{\\mathrm{all}} - \\Sigma_i\\, \\delta_i$   (ATAC)\n"
                  "(super-additive < 0 < sub-additive)", fontsize=9)
    ax.set_title("Cooperativity per enhancer\n(motif count in brackets)",
                 fontweight="bold", fontsize=10)
    return ax


def plot_cooperativity_violin(
    coop: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Cooperativity distribution by number of motifs, each group tested against zero.

    Wilcoxon signed-rank test (t-test for small groups), BH corrected across groups.

    Args:
        coop: build_cooperativity_table output.
        ax: Axis to draw on; created when None.

    Returns:
        The axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))

    df = coop.dropna(subset=["cooperativity"])
    if df.empty:
        return empty_panel(ax, "No cooperativity scores", "Cooperativity by motif count")

    style_ax(ax)
    groups = sorted(df["n_seqlets"].unique())
    data = [df.loc[df["n_seqlets"] == n, "cooperativity"].to_numpy() for n in groups]
    positions = np.arange(len(groups))

    populated = [i for i, d in enumerate(data) if len(d)]
    if populated:
        parts = ax.violinplot([data[i] for i in populated],
                              positions=positions[populated],
                              showmedians=True, showextrema=False, widths=0.6)
        for body in parts["bodies"]:
            body.set_facecolor(PAL["atac"])
            body.set_alpha(0.45)
            body.set_edgecolor("white")
        parts["cmedians"].set_colors("white")
        parts["cmedians"].set_linewidth(2)

    rng = np.random.default_rng(42)
    for i, values in enumerate(data):
        if not len(values):
            continue
        jitter = rng.uniform(-0.07, 0.07, size=len(values))
        ax.scatter(positions[i] + jitter, values, color=PAL["atac"], s=28,
                   alpha=0.85, edgecolors="white", lw=0.4, zorder=3)

    ax.axhline(0, color="gray", lw=1, ls="--", zorder=0)

    # Group tests, then adjust across the groups actually tested.
    raw = {}
    for i, values in enumerate(data):
        if len(values) < 3:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if len(values) >= 10:
                raw[i] = stats.wilcoxon(values).pvalue
            else:
                raw[i] = stats.ttest_1samp(values, 0.0).pvalue

    if raw:
        keys = list(raw)
        pvals = np.array([raw[k] for k in keys])
        order = np.argsort(pvals)
        adj = np.minimum.accumulate(
            (pvals[order] * len(pvals) / np.arange(1, len(pvals) + 1))[::-1]
        )[::-1]
        qvals = dict(zip(np.array(keys)[order], np.clip(adj, 0, 1)))

        y_top = max(d.max() for d in data if len(d))
        y_rng = max(y_top - min(d.min() for d in data if len(d)), 1e-9)
        for i, q in qvals.items():
            ax.text(positions[i], y_top + 0.06 * y_rng, stars(q) or "ns",
                    ha="center", va="bottom", fontsize=10,
                    color=PAL["ink"] if stars(q) else PAL["neut"])

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{n} motifs\n(n={len(d)})" for n, d in zip(groups, data)],
                       fontsize=8)
    ax.set_ylabel("$I = \\delta_{\\mathrm{all}} - \\Sigma_i\\, \\delta_i$   (ATAC)\n"
                  "(super-additive < 0 < sub-additive)", fontsize=9)
    ax.set_title("Cooperativity by motif count", fontweight="bold", fontsize=10)
    return ax


def plot_interaction_volcano(
    pairs: pd.DataFrame,
    ax: plt.Axes | None = None,
    qval_col: str = "qval",
    alpha: float = 0.05,
    label_top: int = 10,
    label_col: str | None = None,
) -> plt.Axes:
    """Volcano plot of motif-pair interactions against adjusted p-value.

    Args:
        pairs: build_pairwise_interactions output after BH correction.
        ax: Axis to draw on; created when None.
        qval_col: Adjusted p-value column.
        alpha: Significance threshold line.
        label_top: Number of pairs labelled per direction.
        label_col: Label column; defaults to enhancer plus motif pair.

    Returns:
        The axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5))

    if pairs.empty or qval_col not in pairs.columns:
        return empty_panel(ax, "No corrected pairwise tests", "Pairwise interactions")

    df = pairs.dropna(subset=["interaction", qval_col]).copy()
    if df.empty:
        return empty_panel(ax, "No corrected pairwise tests", "Pairwise interactions")

    style_ax(ax)
    df["neglog_q"] = -np.log10(df[qval_col].clip(lower=1e-300))
    hit = df[qval_col] < alpha

    colors = np.where(~hit, PAL["neut"],
                      np.where(df["interaction"] < 0, PAL["supra"], PAL["sub"]))
    ax.scatter(df["interaction"], df["neglog_q"], c=colors, s=45,
               alpha=0.85, edgecolors="white", lw=0.5, zorder=3)

    ax.axhline(-np.log10(alpha), color="gray", lw=1, ls="--", zorder=0)
    ax.axvline(0, color="gray", lw=0.8, ls=":", zorder=0)

    # The same factor pair recurs across enhancers, so a bare motif pair does
    # not identify which perturbation a point came from.
    pair_label = df["motif_a"].astype(str) + "\u00d7" + df["motif_b"].astype(str)
    if label_col and label_col in df.columns:
        df["_label"] = df[label_col].astype(str)
    elif "enhancer" in df.columns:
        df["_label"] = df["enhancer"].astype(str) + "\n" + pair_label
    else:
        df["_label"] = pair_label

    # Label each direction separately. Ranking by absolute value alone lets
    # whichever side carries the larger effects take every label, which hides
    # the other half of the result.
    if label_top and hit.any():
        significant = df[hit]
        chosen = pd.concat([
            significant[significant["interaction"] < 0]
            .nsmallest(label_top, "interaction"),
            significant[significant["interaction"] > 0]
            .nlargest(label_top, "interaction"),
        ])
        if not chosen.empty:
            place_labels(ax, chosen["interaction"], chosen["neglog_q"],
                         chosen["_label"], fontsize=6.0)

    ax.text(0.03, 0.96,
            f"{int(hit.sum())} of {len(df)} pairs at FDR < {alpha:g}",
            transform=ax.transAxes, va="top", fontsize=8, color="#444")
    ax.text(0.03, 0.04, "super-additive", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=9, color=PAL["supra"], style="italic")
    ax.text(0.97, 0.04, "sub-additive", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=9, color=PAL["sub"], style="italic")

    ax.set_xlabel("$I = \\delta_{AB} - \\delta_A - \\delta_B$   (ATAC)", fontsize=9)
    ax.set_ylabel("$-\\log_{10}$ FDR", fontsize=9)
    ax.set_title("Pairwise motif interactions", fontweight="bold", fontsize=10)
    return ax


def plot_interaction_heatmap(
    pairs: pd.DataFrame,
    ax: plt.Axes | None = None,
    min_observations: int = 2,
    annotate: bool = True,
) -> plt.Axes:
    """Heatmap of mean interaction per factor pair, averaged over enhancers.

    Args:
        pairs: build_pairwise_interactions output.
        ax: Axis to draw on; created when None.
        min_observations: Minimum co-occurrences to show a cell.
        annotate: Print mean and count in each cell.

    Returns:
        The axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5.5))

    if pairs.empty:
        return empty_panel(ax, "No pairwise interactions", "Interaction grammar")

    df = pairs.copy()
    df["tf_a"] = df["motif_a"].str.split("#").str[0]
    df["tf_b"] = df["motif_b"].str.split("#").str[0]
    # Order the pair so that A-B and B-A land in the same cell.
    ordered = df.apply(lambda r: tuple(sorted([r["tf_a"], r["tf_b"]])), axis=1)
    df["tf_1"] = [p[0] for p in ordered]
    df["tf_2"] = [p[1] for p in ordered]

    agg = (
        df.groupby(["tf_1", "tf_2"])["interaction"]
        .agg(mean="mean", n="size").reset_index()
    )
    agg = agg[agg["n"] >= min_observations]
    if agg.empty:
        return empty_panel(ax, f"No factor pair seen {min_observations}+ times",
                      "Interaction grammar")

    factors = sorted(set(agg["tf_1"]) | set(agg["tf_2"]))
    mat = pd.DataFrame(np.nan, index=factors, columns=factors)
    counts = pd.DataFrame(np.nan, index=factors, columns=factors)
    for _, r in agg.iterrows():
        mat.loc[r["tf_1"], r["tf_2"]] = mat.loc[r["tf_2"], r["tf_1"]] = r["mean"]
        counts.loc[r["tf_1"], r["tf_2"]] = counts.loc[r["tf_2"], r["tf_1"]] = r["n"]

    style_ax(ax)
    vmax = np.nanmax(np.abs(mat.to_numpy()))
    image = ax.imshow(mat.to_numpy(), cmap=CMAP_INTERACTION, vmin=-vmax, vmax=vmax,
                      aspect="equal")

    ax.set_xticks(range(len(factors)))
    ax.set_xticklabels(factors, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(factors)))
    ax.set_yticklabels(factors, fontsize=8)

    if annotate:
        for i, j in itertools.product(range(len(factors)), repeat=2):
            value = mat.to_numpy()[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}\n({int(counts.to_numpy()[i, j])})",
                        ha="center", va="center", fontsize=6,
                        color=PAL["ink"])

    bar = plt.colorbar(image, ax=ax, shrink=0.6, pad=0.03)
    bar.set_label("Mean $I$   (ATAC)\n(super-additive < 0 < sub-additive)",
                  fontsize=8)
    ax.set_title(f"Interaction grammar (pairs seen $\\geq$ {min_observations}$\\times$)",
                 fontweight="bold", fontsize=10, y=1.04)
    return ax


def plot_interaction_vs_spacing(
    pairs: pd.DataFrame,
    ax: plt.Axes | None = None,
    qval_col: str = "qval",
    alpha: float = 0.05,
) -> plt.Axes:
    """Interaction against the edge-to-edge gap between the two motifs.

    Args:
        pairs: build_pairwise_interactions output with spacing attached.
        ax: Axis to draw on; created when None.
        qval_col: Adjusted p-value column, for colouring.
        alpha: Significance threshold, for colouring.

    Returns:
        The axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    if pairs.empty or "spacing_bp" not in pairs.columns:
        return empty_panel(ax, "No spacing information", "Interaction versus spacing")

    df = pairs.dropna(subset=["spacing_bp", "interaction"])
    if len(df) < 3:
        return empty_panel(ax, "Too few pairs", "Interaction versus spacing")

    style_ax(ax)
    if qval_col in df.columns:
        hit = df[qval_col] < alpha
        colors = np.where(hit, PAL["atac"], PAL["neut"])
    else:
        colors = PAL["atac"]

    ax.scatter(df["spacing_bp"], df["interaction"], c=colors, s=45,
               alpha=0.8, edgecolors="white", lw=0.5, zorder=3)
    ax.axhline(0, color="gray", lw=0.8, ls="--", zorder=0)

    rho, p = stats.spearmanr(df["spacing_bp"], df["interaction"])
    pstr = f"p = {p:.1e}" if p < 0.001 else f"p = {p:.3f}"
    ax.text(0.95, 0.95, f"Spearman $\\rho$ = {rho:.2f},  {pstr}\nn = {len(df)} pairs",
            transform=ax.transAxes, va="top", ha = "right",  fontsize=8, color="#444")

    ax.set_xlabel("Edge-to-edge motif spacing (bp)", fontsize=9)
    ax.set_ylabel("$I$   (ATAC)", fontsize=9)
    ax.set_title("Distance dependence", fontweight="bold", fontsize=10)
    return ax


def plot_saturation_diagnostic(
    pairs: pd.DataFrame,
    ax: plt.Axes | None = None,
    against: str = "additive",
    qval_col: str = "qval",
    alpha: float = 0.05,
) -> plt.Axes:
    """Interaction against the additive prediction or the baseline, to check for floor and ceiling artefacts.

    Args:
        pairs: build_pairwise_interactions output.
        ax: Axis to draw on; created when None.
        against: "additive" (floor check) or "baseline" (ceiling check).
        qval_col: Adjusted p-value column, for colouring.
        alpha: Significance threshold, for colouring.

    Returns:
        The axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4))

    if against == "additive":
        if pairs.empty or not {"delta_a", "delta_b"}.issubset(pairs.columns):
            return empty_panel(ax, "No main effects available", "Saturation check")
        x = (pairs["delta_a"] + pairs["delta_b"]).to_numpy()
        xlabel = "Additive prediction  $\\delta_A + \\delta_B$"
        note = "floor artefact would slope downwards"
    elif against == "baseline":
        if pairs.empty or "baseline" not in pairs.columns:
            return empty_panel(ax, "No baseline column", "Saturation check")
        x = pairs["baseline"].to_numpy()
        xlabel = "Unperturbed accessibility  $y_0$"
        note = "ceiling artefact would slope downwards"
    else:
        raise ValueError(f"Unknown diagnostic '{against}'.")

    y = pairs["interaction"].to_numpy()
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3:
        return empty_panel(ax, "Too few pairs", "Saturation check")

    style_ax(ax)
    if qval_col in pairs.columns:
        hit = pairs.loc[keep, qval_col].to_numpy() < alpha
        colors = np.where(hit, PAL["atac"], PAL["neut"])
    else:
        colors = PAL["atac"]

    ax.scatter(x, y, c=colors, s=42, alpha=0.8, edgecolors="white",
               lw=0.5, zorder=3)
    ax.axhline(0, color="gray", lw=0.8, ls="--", zorder=0)

    slope, intercept, r, p, _ = stats.linregress(x, y)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, intercept + slope * xs, color=PAL["ink"], lw=1.2, zorder=2)

    pstr = f"p = {p:.1e}" if p < 0.001 else f"p = {p:.3f}"
    ax.text(0.04, 0.05, f"r = {r:.2f},  {pstr}\n{note}",
            transform=ax.transAxes, va="bottom", fontsize=7.5, color="#444")

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("$I$   (ATAC)", fontsize=9)
    ax.set_title(f"Saturation check ({against})", fontweight="bold", fontsize=10)
    return ax


def plot_specificity_control(
    motifs: pd.DataFrame,
    controls: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Distribution of motif scramble effects against control-window effects.

    Args:
        motifs: build_motif_effects output of the motif run.
        controls: build_motif_effects output of the control run.
        ax: Axis to draw on; created when None.

    Returns:
        The axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))

    if motifs.empty or controls is None or controls.empty:
        return empty_panel(ax, "Control run not available", "Motif specificity")

    groups = [controls["delta"].dropna().to_numpy(),
              motifs["delta"].dropna().to_numpy()]
    names = [f"Control windows\n(n={len(groups[0])})",
             f"Called motifs\n(n={len(groups[1])})"]
    colors = [PAL["neut"], PAL["atac"]]

    style_ax(ax)
    positions = np.arange(2)
    parts = ax.violinplot(groups, positions=positions, showmedians=True,
                          showextrema=False, widths=0.65)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.45)
        body.set_edgecolor("white")
    parts["cmedians"].set_colors("white")
    parts["cmedians"].set_linewidth(2)

    rng = np.random.default_rng(11)
    for i, (values, color) in enumerate(zip(groups, colors)):
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        ax.scatter(positions[i] + jitter, values, color=color, s=18,
                   alpha=0.7, edgecolors="white", lw=0.3, zorder=3)

    ax.axhline(0, color="gray", lw=1, ls="--", zorder=0)

    # The control distribution defines what counts as an unusually large loss.
    threshold = np.percentile(groups[0], 5)
    ax.axhline(threshold, color=PAL["sub"], lw=1, ls=":", zorder=1)
    beyond = float((groups[1] < threshold).mean())

    stat, p = stats.mannwhitneyu(groups[1], groups[0], alternative="less")
    pstr = f"p = {p:.1e}" if p < 0.001 else f"p = {p:.3f}"
    ax.text(0.5, 0.04,
            f"Mann-Whitney {pstr}\n"
            f"{beyond:.0%} of motifs below the control 5th percentile",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7.5, color="#444")

    ax.set_xticks(positions)
    ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel("$\\delta_i = \\log(y_i / y_0)$   (ATAC)", fontsize=9)
    ax.set_title("Motif specificity control", fontweight="bold", fontsize=10)
    return ax


def plot_atac_rna_concordance(
    coop_atac: pd.DataFrame,
    coop_rna: pd.DataFrame,
    ax: plt.Axes | None = None,
    label_points: bool = True,
    label_top: int = 10,
) -> plt.Axes:
    """Per-enhancer cooperativity on accessibility against expression.

    Args:
        coop_atac: Cooperativity table on ATAC.
        coop_rna: Cooperativity table on RNA.
        ax: Axis to draw on; created when None.
        label_points: Annotate enhancer names.
        label_top: Number of enhancers labelled, ranked by distance from the origin.

    Returns:
        The axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4.5))

    merged = coop_atac.merge(coop_rna, on="enhancer", suffixes=("_atac", "_rna"))
    merged = merged.dropna(subset=["cooperativity_atac", "cooperativity_rna"])
    if len(merged) < 3:
        return empty_panel(ax, "Too few paired enhancers", "ATAC versus RNA")

    style_ax(ax)
    x = merged["cooperativity_atac"].to_numpy()
    y = merged["cooperativity_rna"].to_numpy()

    ax.axhline(0, color="gray", lw=0.8, ls="--", zorder=0)
    ax.axvline(0, color="gray", lw=0.8, ls="--", zorder=0)

    concordant = np.sign(x) == np.sign(y)
    colors = np.where(concordant,
                      np.where(x < 0, PAL["supra"], PAL["sub"]), PAL["neut"])
    ax.scatter(x, y, c=colors, s=60, edgecolors="white", lw=0.6, zorder=3)

    if label_points and label_top:
        names = merged["enhancer"].to_numpy()
        chosen = np.argsort(-np.hypot(x, y))[:label_top]
        place_labels(ax, x[chosen], y[chosen], names[chosen])

    rho, p = stats.spearmanr(x, y)
    pstr = f"p = {p:.1e}" if p < 0.001 else f"p = {p:.3f}"
    ax.text(0.04, 0.96,
            f"Spearman $\\rho$ = {rho:.2f},  {pstr}\n"
            f"{int(concordant.sum())}/{len(merged)} concordant in sign",
            transform=ax.transAxes, va="top", fontsize=8, color="#444")

    ax.set_xlabel("$I$   (ATAC)", fontsize=9)
    ax.set_ylabel("$I$   (RNA)", fontsize=9)
    ax.set_title("Propagation to target gene", fontweight="bold", fontsize=10)
    return ax


# --------------------------------------------------------------------------- #
# Figure assembly
# --------------------------------------------------------------------------- #

def make_main_figure(
    coop_atac: pd.DataFrame,
    coop_rna: pd.DataFrame,
    pairs: pd.DataFrame,
    motifs: pd.DataFrame,
    *,
    controls: pd.DataFrame | None = None,
    figsize: tuple[float, float] = (16, 13),
) -> plt.Figure:
    """Assemble the six-panel within-enhancer main figure.

    Args:
        coop_atac: Cooperativity table on ATAC.
        coop_rna: Cooperativity table on RNA.
        pairs: Pairwise interactions after BH correction.
        motifs: Motif effects.
        controls: Control-window effects, shaded in panel A.
        figsize: Figure size in inches.

    Returns:
        The figure.
    """
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("#FAFAF8")
    grid = GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.30,
                    left=0.08, right=0.96, top=0.92, bottom=0.06)

    axes = [fig.add_subplot(grid[i, j]) for i in range(3) for j in range(2)]

    plot_motif_importance(motifs, ax=axes[0], top_n=10, controls=controls)
    #plot_cooperativity_forest(coop_atac, ax=axes[2], top_n=20)
    plot_interaction_volcano(pairs, ax=axes[1])
    plot_interaction_heatmap(pairs, ax=axes[2])
    plot_interaction_vs_spacing(pairs, ax=axes[3])
    plot_additivity(coop_atac, ax=axes[4], readout="ATAC")
    plot_atac_rna_concordance(coop_atac, coop_rna, ax=axes[5])

    for ax, letter in zip(axes, "ABCDEF"):
        ax.text(-0.14, 1.05, letter, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="bottom")

    #fig.suptitle("Transcription factor motif cooperativity within K562 enhancers",
    #             fontsize=14, fontweight="bold", y=0.96)
    return fig


def plot_saturation_split(
    split: pd.DataFrame,
    ax: plt.Axes | None = None,
    show_full: bool = True,
) -> plt.Axes:
    """Saturation check with x and y estimated from disjoint shuffle halves.

    Args:
        split: build_saturation_split output.
        ax: Axis to draw on; created when None.
        show_full: Also print the full-data (coupled) correlation.

    Returns:
        The axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4))

    if split.empty or "additive_split" not in split.columns:
        return empty_panel(ax, "Run tf_cooperativity.build_saturation_split",
                           "Saturation check (split)")

    df = split.dropna(subset=["additive_split", "interaction_split"])
    if len(df) < 3:
        return empty_panel(ax, "Too few pairs", "Saturation check (split)")

    style_ax(ax)
    x = df["additive_split"].to_numpy()
    y = df["interaction_split"].to_numpy()

    ax.scatter(x, y, c=PAL["atac"], s=42, alpha=0.8, edgecolors="white",
               lw=0.5, zorder=3)
    ax.axhline(0, color="gray", lw=0.8, ls="--", zorder=0)

    lines = []
    if len(np.unique(x)) > 1:
        slope, intercept, r, p, _ = stats.linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, intercept + slope * xs, color=PAL["ink"], lw=1.2, zorder=2)
        lines.append(f"Pearson  r = {r:+.2f} {pformat(p)}")

        # Rank correlation is far less sensitive to a handful of extreme pairs.
        rho, p_rho = stats.spearmanr(x, y)
        lines.append(f"Spearman r = {rho:+.2f} {pformat(p_rho)}")

        # The baseline enters the two axes with opposite signs, so it induces a
        # negative correlation of its own that splitting cannot remove.
        if "log_baseline" in df.columns:
            r_par, p_par, _ = partial_correlation(x, y, df["log_baseline"])
            if np.isfinite(r_par):
                lines.append(f"partial  r = {r_par:+.2f} {pformat(p_par)}"
                             f"  | $y_0$")

    if show_full and {"additive_full", "interaction_full"}.issubset(df.columns):
        full = df.dropna(subset=["additive_full", "interaction_full"])
        if len(full) > 2 and len(np.unique(full["additive_full"])) > 1:
            r_full = stats.pearsonr(full["additive_full"],
                                    full["interaction_full"])[0]
            lines.append(f"coupled  r = {r_full:+.2f}  (full data)")

    ax.text(0.03, 0.04, "\n".join(lines) or "no trend estimable",
            transform=ax.transAxes, va="bottom", fontsize=6.8, color="#444",
            family="monospace")

    ax.set_xlabel("$\\delta_A + \\delta_B$   (shuffles 1 to 5)", fontsize=9)
    ax.set_ylabel("$I$   (shuffles 6 to 10)", fontsize=9)
    ax.set_title("Saturation check, decoupled", fontweight="bold", fontsize=10)
    return ax


def plot_floor_check(
    floor_df: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Observed double-scramble readout against the multiplicative null; the band marks the readout floor.

    Args:
        floor_df: build_floor_check output.
        ax: Axis to draw on; created when None.

    Returns:
        The axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4))

    if floor_df.empty or "null_prediction" not in floor_df.columns:
        return empty_panel(ax, "Run tf_cooperativity.build_floor_check",
                           "Floor check")

    df = floor_df.dropna(subset=["null_prediction", "observed"])
    df = df[(df["null_prediction"] > 0) & (df["observed"] > 0)]
    if len(df) < 3:
        return empty_panel(ax, "Too few pairs", "Floor check")

    style_ax(ax)
    floor = float(df["floor"].iloc[0])
    x = df["null_prediction"].to_numpy()
    y = df["observed"].to_numpy()

    lim = [min(x.min(), y.min(), floor) * 0.6, max(x.max(), y.max()) * 1.6]
    ax.plot(lim, lim, color="gray", lw=1, ls="--", zorder=1)
    ax.axvspan(lim[0], floor, color=PAL["sub"], alpha=0.16, zorder=0)
    ax.axhline(floor, color=PAL["sub"], lw=1, ls=":", zorder=1)

    below = df["null_below_floor"].to_numpy()
    ax.scatter(x, y, c=np.where(below, PAL["sub"], PAL["atac"]), s=42,
               alpha=0.85, edgecolors="white", lw=0.5, zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)

    ax.text(0.04, 0.95,
            f"floor = {floor:.3g}\n"
            f"{int(below.sum())}/{len(df)} pairs predicted below it",
            transform=ax.transAxes, va="top", fontsize=7.5, color="#444")

    ax.set_xlabel("Multiplicative null  $y_0 e^{\\delta_A + \\delta_B}$", fontsize=9)
    ax.set_ylabel("Observed  $y_{AB}$", fontsize=9)
    ax.set_title("Floor check", fontweight="bold", fontsize=10)
    return ax


def plot_dose_response(
    dose: pd.DataFrame,
    ax: plt.Axes | None = None,
    max_order: int | None = None,
) -> plt.Axes:
    """Observed effect against the additive null for every combination order.

    Args:
        dose: build_dose_response output.
        ax: Axis to draw on; created when None.
        max_order: Highest order shown.

    Returns:
        The axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4))

    if dose.empty or "predicted" not in dose.columns:
        return empty_panel(ax, "Run tf_cooperativity.build_dose_response",
                           "Dose response")

    df = dose.dropna(subset=["predicted", "observed"])
    if max_order:
        df = df[df["n_scrambled"] <= max_order]
    if len(df) < 3:
        return empty_panel(ax, "Too few combinations", "Dose response")

    style_ax(ax)
    orders = sorted(df["n_scrambled"].unique())
    shades = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(orders), 2)))

    lim = np.array([df[["predicted", "observed"]].min().min(),
                    df[["predicted", "observed"]].max().max()])
    pad = 0.08 * (lim[1] - lim[0] or 1.0)
    lim = lim + np.array([-pad, pad])
    ax.plot(lim, lim, color="gray", lw=1, ls="--", zorder=1)

    for shade, order in zip(shades, orders):
        sub = df[df["n_scrambled"] == order]
        ax.scatter(sub["predicted"], sub["observed"], color=shade, s=26,
                   alpha=0.75, edgecolors="white", lw=0.3, zorder=3,
                   label=f"{order} scrambled")

    # Binned median traces the shape of any departure from the diagonal.
    edges = np.linspace(df["predicted"].min(), df["predicted"].max(), 9)
    centres, medians = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        window = df[(df["predicted"] >= lo) & (df["predicted"] < hi)]
        if len(window) >= 3:
            centres.append(0.5 * (lo + hi))
            medians.append(window["observed"].median())
    if len(centres) > 2:
        ax.plot(centres, medians, color=PAL["sub"], lw=2, zorder=4,
                label="binned median")

    shortfall = df[df["n_scrambled"] >= 2].groupby("n_scrambled")["deviation"].mean()
    summary = "  ".join(f"{int(k)}: {v:+.2f}" for k, v in shortfall.items())
    ax.text(0.03, 0.96, f"mean deviation by order\n{summary}",
            transform=ax.transAxes, va="top", fontsize=6.8, color="#444",
            family="monospace")

    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.legend(fontsize=6, loc="lower right", framealpha=0.85)
    ax.set_xlabel("Additive null  $\\Sigma_i\\, \\delta_i$", fontsize=9)
    ax.set_ylabel("Observed  $\\delta_{\\mathrm{combo}}$", fontsize=9)
    ax.set_title("Dose response across orders", fontweight="bold", fontsize=10)
    return ax


def make_supplementary_figure(
    motifs: pd.DataFrame,
    coop_atac: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    controls: pd.DataFrame | None = None,
    split: pd.DataFrame | None = None,
    floor_df: pd.DataFrame | None = None,
    dose: pd.DataFrame | None = None,
    figsize: tuple[float, float] = (15, 13),
) -> plt.Figure:
    """Assemble the nine-panel within-enhancer supplementary figure.

    Panels with missing inputs are drawn as placeholders.

    Args:
        motifs: Motif effects with attribution.
        coop_atac: Cooperativity table on ATAC.
        pairs: Pairwise interactions with spacing.
        controls: Control-window effects.
        split: build_saturation_split output.
        floor_df: build_floor_check output.
        dose: build_dose_response output.
        figsize: Figure size in inches.

    Returns:
        The figure.
    """
    fig, axes = plt.subplots(4, 2, figsize=figsize)
    fig.patch.set_facecolor("#FAFAF8")
    flat = axes.ravel()
    blank = pd.DataFrame()

    plot_effect_vs_attribution(motifs, ax=flat[1])
    plot_specificity_control(motifs, controls, ax=flat[0])
    plot_cooperativity_violin(coop_atac, ax=flat[2])

    #plot_interaction_vs_spacing(pairs, ax=flat[3]) #moved to main panel
    plot_saturation_diagnostic(pairs, ax=flat[3], against="additive")
    plot_saturation_split(split if split is not None else blank, ax=flat[4])

    plot_saturation_diagnostic(pairs, ax=flat[6], against="baseline")
    plot_floor_check(floor_df if floor_df is not None else blank, ax=flat[5])
    #plot_dose_response(dose if dose is not None else blank, ax=flat[8])

    # Slicing out the hidden axis prevents the lettering utility from labeling blank space
    flat[7].set_visible(False)
    panel_letters(flat[:7], dx=-0.18)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.suptitle("Supplementary: validation, specificity and artefact checks",
                 fontsize=12, fontweight="bold", y=0.985)
    return fig