"""Tests for ``element_epistasis.cross_level`` against synthetic data with known answers.

Each block builds input whose correct output is known by construction, so a
failure names the broken assumption rather than merely reporting a number that
looks wrong. Run with ``python test_cross_level.py``.
"""

import numpy as np
import pandas as pd

from element_epistasis import cross_level as cl

RNG = np.random.default_rng(0)
PASS, FAIL = [], []


def check(condition, message):
    """Record one assertion."""
    (PASS if condition else FAIL).append(message)
    print(("  ok   " if condition else "  FAIL ") + message)


# --------------------------------------------------------------------------- #
print("\n1. normalise_measured_pairs, pair_name shape")
# --------------------------------------------------------------------------- #

exp = pd.DataFrame({
    "pair_name": ["ANXA1_enh1+enh3", "MGAT1_enh2+enh2-3", "ANXA1_enh1+enh3"],
    "interaction_term": [-0.04, 0.21, -0.06],
    "p_value": [0.26, 0.01, 0.90],
    "dataset": ["JB_ST_002", "JB_ST_002", "JB_ST_004"],
})
norm = cl.normalise_measured_pairs(exp, collapse="most_significant")

check(len(norm) == 2, "duplicate pair collapsed to one row")
row = norm[norm["enhA"] == "ANXA1_enh1"].iloc[0]
check(row["enhB"] == "ANXA1_enh3", "bare suffix qualified with the gene")
check(np.isclose(row["interaction_beta"], -0.04),
      "most significant record retained, not the first")
check(row["n_records"] == 2, "record count preserved")
mgat = norm[norm["gene"] == "MGAT1"].iloc[0]
check(mgat["enhB"] == "MGAT1_enh2-3",
      "compound suffix enh2-3 parsed without truncation")

mean_collapse = cl.normalise_measured_pairs(exp, collapse="mean")
row = mean_collapse[mean_collapse["enhA"] == "ANXA1_enh1"].iloc[0]
check(np.isclose(row["interaction_beta"], -0.05), "mean collapse averages")

# --------------------------------------------------------------------------- #
print("\n2. normalise_measured_pairs, enhA/enhB shape and pair ordering")
# --------------------------------------------------------------------------- #

adj = pd.DataFrame({
    "enhA": ["ANXA1_enh3", "GYPC_enh1"],
    "enhB": ["ANXA1_enh1", "GYPC_enh2"],
    "interaction_beta": [0.07, -0.02],
    "source_file": ["JB_ST_002", "JB_ST_002"],
    "": [0.035, 0.4],
})
norm_adj = cl.normalise_measured_pairs(adj)
row = norm_adj[norm_adj["gene"] == "ANXA1"].iloc[0]
check(row["enhA"] == "ANXA1_enh1" and row["enhB"] == "ANXA1_enh3",
      "pair sorted so that join order does not matter")
check(np.isclose(row["measured_pval"], 0.035),
      "trailing unnamed column read as the p-value")

# --------------------------------------------------------------------------- #
print("\n3. link_motif_pairs_to_enhancers joins on the unordered pair")
# --------------------------------------------------------------------------- #

pair_agg = pd.DataFrame({
    "gene": ["ANXA1", "GYPC"],
    "enhA": ["ANXA1_enh1", "GYPC_enh1"],
    "enhB": ["ANXA1_enh3", "GYPC_enh2"],
    "enh_pair": ["ANXA1_enh1_x_ANXA1_enh3", "GYPC_enh1_x_GYPC_enh2"],
    "motif_mean": [-0.01, -0.025],
    "motif_strongest": [-0.016, -0.035],
    "motif_sum": [-0.06, -0.22],
    "motif_max": [0.002, -0.01],
    "n_motif_pairs": [6, 9],
    "n_significant": [0, 0],
})
# Deliberately reversed order on the enhancer side.
enh_inters = pd.DataFrame({
    "gene": ["ANXA1", "GYPC"],
    "full_a": ["ANXA1_enh3", "GYPC_enh2"],
    "full_b": ["ANXA1_enh1", "GYPC_enh1"],
    "coef": [-0.042, -0.037],
    "qval": [4.4e-6, 3.2e-4],
})
linked = cl.link_motif_pairs_to_enhancers(pair_agg, enh_inters, norm_adj)
check(linked["enh_coef"].notna().all(),
      "enhancer coefficients matched despite reversed pair order")
check(linked["interaction_beta"].notna().all(),
      "measured coefficients matched on the same key")
check(np.isclose(linked.loc[linked["gene"] == "ANXA1", "enh_coef"].iloc[0],
                 -0.042), "correct coefficient attached to the correct pair")

# --------------------------------------------------------------------------- #
print("\n4. summarise_within_gene_link separates the two routes")
# --------------------------------------------------------------------------- #

# Construct a case with a strong between-gene relationship and no within-gene
# one: gene means lie on a line, but the deviations inside each gene are
# independent noise. A correct implementation reports pooled high, within zero.
genes, xs, ys = [], [], []
for g, level in enumerate(np.linspace(-1.0, 1.0, 10)):
    for _ in range(5):
        genes.append(f"G{g}")
        xs.append(level + RNG.normal(0, 0.05))
        ys.append(level + RNG.normal(0, 0.05))
pure_between = pd.DataFrame({"gene": genes, "motif_mean": xs, "enh_coef": ys})
res = cl.summarise_within_gene_link(pure_between)

check(res["pooled_pearson_r"] > 0.9, "pooled correlation is high as constructed")
check(abs(res["within_pearson_r"]) < 0.3,
      "within-gene correlation collapses when the signal is between genes")
check(res["between_pearson_r"] > 0.9, "between-gene view recovers the signal")
check(res["n_genes_kept"] == 10 and res["n_pairs_kept"] == 50,
      "gene and pair counts correct")

# The converse: identical gene means, real within-gene structure.
genes, xs, ys = [], [], []
for g in range(10):
    deviations = RNG.normal(0, 0.3, 5)
    for d in deviations:
        genes.append(f"G{g}")
        xs.append(d)
        ys.append(d + RNG.normal(0, 0.05))
pure_within = pd.DataFrame({"gene": genes, "motif_mean": xs, "enh_coef": ys})
res_w = cl.summarise_within_gene_link(pure_within)
check(res_w["within_pearson_r"] > 0.9,
      "within-gene correlation survives when the signal is within genes")
check(res_w["per_gene_n"] == 10, "per-gene rank correlations computed")

# --------------------------------------------------------------------------- #
print("\n5. leverage diagnostic detects a single driving point")
# --------------------------------------------------------------------------- #

x = np.concatenate([RNG.normal(0, 0.01, 40), [1.0]])
y = np.concatenate([RNG.normal(0, 0.01, 40), [1.0]])
jack = cl._jackknife_influence(x, y)
check(jack["jack_r_full"] > 0.9, "one extreme point creates a high correlation")
check(jack["jack_r_min"] < 0.5,
      "leave-one-out exposes that the correlation rests on that point")

# --------------------------------------------------------------------------- #
print("\n6. aggregate_motifs_per_enhancer")
# --------------------------------------------------------------------------- #

motifs = pd.DataFrame({
    "enhancer": ["E1", "E1", "E1", "E2", "E2"],
    "motif": ["GATA2", "KLF1", "FOS", "SP5", "FLI1"],
    "delta": [-1.5, -0.4, 0.1, -0.2, -0.3],
    "qval": [1e-9, 0.01, 0.8, 0.02, 0.03],
})
coop = pd.DataFrame({"enhancer": ["E1", "E2"], "cooperativity": [0.9, -0.2],
                     "n_seqlets": [3, 2]})
agg = cl.aggregate_motifs_per_enhancer(motifs, coop)
e1 = agg[agg["enhancer"] == "E1"].iloc[0]
check(np.isclose(e1["motif_strongest"], -1.5), "strongest is the most negative")
check(np.isclose(e1["motif_max_abs"], 1.5), "max absolute effect correct")
check(np.isclose(e1["motif_sum_abs"], 2.0), "summed absolute effect correct")
check(e1["n_motifs"] == 3 and e1["n_significant"] == 2,
      "motif and significance counts correct")
check(np.isclose(e1["cooperativity"], 0.9), "cooperativity joined")

# --------------------------------------------------------------------------- #
print("\n7. link_motif_effects_to_enhancers reports unmatched enhancers")
# --------------------------------------------------------------------------- #

enh_mains = pd.DataFrame({
    "gene": ["E1"], "full_a": ["E1"], "coef": [-0.31], "qval": [1e-4],
})
measured = cl.normalise_measured_singles(
    pd.DataFrame({"enhancer": ["E1", "E2"], "beta": [-0.28, -0.05],
                  "source_dataset": ["JB_ST_004", "JB_ST_004"]}))
import warnings as _w
with _w.catch_warnings(record=True) as caught:
    _w.simplefilter("always")
    ml = cl.link_motif_effects_to_enhancers(agg, enh_mains, measured)
check(any("no enhancer-level coefficient" in str(c.message) for c in caught),
      "unmatched enhancers raise a warning rather than disappearing")
check(ml["measured_beta"].notna().all(), "measured betas attached to both rows")
check(int(ml["enh_coef"].isna().sum()) == 1, "E2 correctly left unmatched")

# --------------------------------------------------------------------------- #
print("\n8. summarise_spacing_partial recovers a known partial correlation")
# --------------------------------------------------------------------------- #

# Ground truth: the interaction depends only on the summed main effects, and
# spacing is correlated with those main effects but has no direct influence.
# The raw spacing correlation must therefore be non-zero and the partial
# correlation must be approximately zero.
n = 300
additive = RNG.normal(-1.0, 0.5, n)
spacing = 40 + 25 * additive + RNG.normal(0, 5, n)
interaction = -0.5 * additive + RNG.normal(0, 0.05, n)
fake_pairs = pd.DataFrame({
    "enhancer": ["E"] * n, "motif_a": ["A"] * n, "motif_b": ["B"] * n,
    "interaction": interaction, "spacing_bp": spacing,
    "delta_a": additive / 2, "delta_b": additive / 2,
})
sp = cl.summarise_spacing_partial(fake_pairs, log_spacing=False)
check(abs(sp["raw_pearson_r"]) > 0.5,
      "raw spacing correlation is strong, as constructed")
check(abs(sp["partial_r"]) < 0.15,
      "partial correlation collapses when spacing has no direct effect")
check(abs(sp["reverse_partial_r"]) > 0.5,
      "reverse partial retains the saturation coupling, which is the real one")

# The converse: spacing genuinely drives the interaction on top of the coupling.
interaction2 = -0.5 * additive + 0.01 * spacing + RNG.normal(0, 0.05, n)
fake_pairs2 = fake_pairs.assign(interaction=interaction2)
sp2 = cl.summarise_spacing_partial(fake_pairs2, log_spacing=False)
check(abs(sp2["partial_r"]) > 0.5,
      "partial correlation survives when spacing has a real direct effect")

check("partial_rho" in sp2, "rank partial computed alongside the Pearson one")

# --------------------------------------------------------------------------- #
print("\n9. summaries render without error")
# --------------------------------------------------------------------------- #

text = cl.format_cross_level_summary(
    pair_link=cl.summarise_pair_link(linked),
    within_gene=res,
    motif_link=cl.summarise_motif_enhancer_link(ml),
    spacing=sp2,
)
check(len(text.splitlines()) > 20, "summary block renders")
check("G1." in text and "G5." in text and "G2." in text and "G3." in text,
      "all four analyses present in the rendered block")

# --------------------------------------------------------------------------- #
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for message in FAIL:
        print("  FAILED:", message)
    raise SystemExit(1)
