"""Shared palette, colormaps, axis style and annotation helpers for all figures.

Sign convention: an interaction I < 0 is labelled super-additive and I > 0
sub-additive. "Synergy" and "redundancy" are not used, because redundant
(shadow) enhancers produce a negative I.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

try:
    from adjustText import adjust_text
    _HAS_ADJUST_TEXT = True
except ImportError:  # labels fall back to fixed offsets
    _HAS_ADJUST_TEXT = False


PAL = dict(
    atac="#2176AE",
    rna="#E84855",
    supra="#06A77D",   # super-additive, I < 0
    sub="#F4A261",     # sub-additive,   I > 0
    neut="#A8AAAD",
    bg="#F7F7F5",
    spine="#CCCCCC",
    ink="#333333",
    crispri="#7B4B94",
)
PAL["syn"] = PAL["supra"]   # retained so older notebook cells keep working
PAL["red"] = PAL["sub"]

CMAP_INTERACTION = LinearSegmentedColormap.from_list(
    "super_sub_additive", [PAL["supra"], "#FFFFFF", PAL["sub"]]
)
CMAP_EFFECT = LinearSegmentedColormap.from_list(
    "effect", [PAL["atac"], "#FFFFFF", PAL["rna"]]
)

# Axis label fragments, defined once so the figures and the methods section
# cannot drift apart.
LBL_DELTA = "$\\delta_i = \\log(y_i / y_0)$"
LBL_BETA_MAIN = "GLM main effect  $\\beta_i$"
LBL_BETA_INTER = "GLM interaction  $\\beta_{ij}$"
LBL_I = "$I = \\delta_{AB} - \\delta_A - \\delta_B$"
LBL_DIRECTION = "(super-additive < 0 < sub-additive)"


def style_ax(ax: plt.Axes) -> plt.Axes:
    """Apply the shared axis style.

    Args:
        ax: Axis to style.

    Returns:
        The styled axis.
    """
    ax.set_facecolor(PAL["bg"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(PAL["spine"])
    ax.tick_params(labelsize=9, color=PAL["spine"])
    return ax


def stars(q: float) -> str:
    """Convert an adjusted p-value into a significance marker.

    Args:
        q: Adjusted p-value.

    Returns:
        "***", "**", "*" for q below 0.001, 0.01 and 0.05, otherwise "".
    """
    if not np.isfinite(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


def pformat(p: float) -> str:
    """Format a p-value for a figure annotation.

    Args:
        p: p-value.

    Returns:
        Formatted string such as "p = 0.012" or "p = 3.4e-05".
    """
    if not np.isfinite(p):
        return "p = n/a"
    return f"p = {p:.1e}" if p < 0.001 else f"p = {p:.3f}"


def direction_color(value: float) -> str:
    """Colour for an interaction value by its sign.

    Args:
        value: Interaction value.

    Returns:
        Super-additive colour for negative values, sub-additive colour otherwise.
    """
    return PAL["supra"] if value < 0 else PAL["sub"]


def label_points(ax: plt.Axes, xs: Sequence[float], ys: Sequence[float],
                 labels: Sequence[str], fontsize: float = 6.5) -> list:
    """Label points on an axis, using adjustText to avoid overlaps when installed.

    Args:
        ax: Axis to draw on.
        xs: x coordinates of the points.
        ys: y coordinates of the points.
        labels: One label per point.
        fontsize: Label font size.

    Returns:
        The created text objects.
    """
    texts = [
        ax.text(x, y, lab, fontsize=fontsize, color=PAL["ink"],
                ha="center", va="bottom")
        for x, y, lab in zip(xs, ys, labels)
    ]
    if _HAS_ADJUST_TEXT and texts:
        adjust_text(texts, ax=ax, expand=(1.2, 1.4),
                    arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.6))
    return texts


def empty_panel(ax: plt.Axes, message: str, title: str) -> plt.Axes:
    """Draw a placeholder panel for missing inputs.

    Args:
        ax: Axis to draw on.
        message: Text shown in the centre of the panel.
        title: Panel title.

    Returns:
        The axis.
    """
    style_ax(ax)
    ax.text(0.5, 0.5, message, ha="center", va="center",
            transform=ax.transAxes, color="gray", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontweight="bold", fontsize=10)
    return ax


def add_direction_annotations(ax: plt.Axes, supra_corner: str = "lower right",
                              sub_corner: str = "upper left") -> plt.Axes:
    """Mark the super-additive and sub-additive sides of a panel.

    Args:
        ax: Axis to annotate.
        supra_corner: Corner for the super-additive label, e.g. "lower right".
        sub_corner: Corner for the sub-additive label.

    Returns:
        The axis.
    """
    corners = {
        "lower right": (0.96, 0.05, "right", "bottom"),
        "lower left": (0.04, 0.05, "left", "bottom"),
        "upper right": (0.96, 0.95, "right", "top"),
        "upper left": (0.04, 0.95, "left", "top"),
    }
    for corner, text, color in [(supra_corner, "super-additive", PAL["supra"]),
                                (sub_corner, "sub-additive", PAL["sub"])]:
        x, y, ha, va = corners[corner]
        ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
                fontsize=9, color=color, style="italic")
    return ax


def panel_letters(axes: Sequence[plt.Axes], letters: str | None = None,
                  dx: float = -0.14, dy: float = 1.05) -> Sequence[plt.Axes]:
    """Add bold panel letters to a sequence of axes.

    Args:
        axes: Axes to label, in order.
        letters: Letters to use. Defaults to "ABCDEFGHIJKL".
        dx: Horizontal offset in axes coordinates.
        dy: Vertical offset in axes coordinates.

    Returns:
        The axes.
    """
    letters = letters or "ABCDEFGHIJKL"
    for ax, letter in zip(axes, letters):
        ax.text(dx, dy, letter, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="bottom")
    return axes