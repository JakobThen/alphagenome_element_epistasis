"""End-to-end check that the documented notebook chain runs and joins correctly.

Builds a synthetic cross-enhancer scramble frame with the schema the real
pipeline produces, pushes it through the actual ``glm`` and
``tf_cooperativity`` builders, and then through the new ``cross_level``
functions. The point is not to recover a known effect size but to verify that
every join key lines up when the real builder functions produce the tables,
rather than when the test writes them by hand.
"""

import itertools
import warnings

import numpy as np
import pandas as pd

import element_epistasis.glm as glm
import element_epistasis.enhancer_analysis as ea
import element_epistasis.tf_cooperativity as tc
import element_epistasis.cross_level as cl

RNG = np.random.default_rng(7)
N_SHUFFLES = 6

GENES = {
    "GENEA": {"enh1": ["GATA2", "KLF1"], "enh2": ["FOS", "SP5"]},
    "GENEB": {"enh1": ["GATA6", "MKX"], "enh3": ["STAT5A", "FLI1"]},
    "GENEC": {"enh1": ["ATF4", "CUX1"], "enh2": ["KLF4"], "enh4": ["MECOM"]},
}
# True per-unit effects on expression, plus one true across-enhancer interaction
# per gene so the aggregation has something to recover.
UNIT_EFFECT = {}
for gene, enhancers in GENES.items():
    for enh, motifs in enhancers.items():
        for motif in motifs:
            UNIT_EFFECT[f"{gene}_{enh}|{motif}"] = -abs(RNG.normal(0.15, 0.08))


def build_scramble_frame():
    """Synthetic combination frame matching the cross-enhancer schema."""
    rows = []
    for gene, enhancers in GENES.items():
        units = [f"{enh}|{motif}" for enh, motifs in enhancers.items()
                 for motif in motifs]
        loci = [f"{gene}_{enh}" for enh in enhancers]
        combos = [()] + [(u,) for u in units] + list(itertools.combinations(units, 2))

        for combo in combos:
            for shuffle in range(N_SHUFFLES):
                log_rna = 6.0
                for unit in combo:
                    log_rna += UNIT_EFFECT[f"{gene}_{unit}"]
                if len(combo) == 2:
                    same = combo[0].split("|")[0] == combo[1].split("|")[0]
                    # Within enhancer sub-additive, across enhancer
                    # super-additive, matching the reported direction.
                    log_rna += 0.05 if same else -0.02

                atac = []
                for locus in loci:
                    log_atac = 5.0
                    for unit in combo:
                        cis = f"{gene}_{unit.split('|')[0]}" == locus
                        log_atac += (-0.6 if cis else -0.03)
                    atac.append(float(np.exp(log_atac + RNG.normal(0, 0.01))))

                rows.append({
                    "name": gene,
                    "scrambled_seqlets": list(combo),
                    "unscrambled_seqlets": units,
                    "n_scrambled": len(combo),
                    "shuffle_idx": shuffle,
                    "RNA_counts": float(np.exp(log_rna + RNG.normal(0, 0.01))),
                    "ATAC_counts": atac,
                    "target_enhancer": loci,
                })
    return pd.DataFrame(rows)


def build_enhancer_level():
    """Synthetic enhancer-level knockout coefficients, one fit per gene."""
    rows = []
    for gene, enhancers in GENES.items():
        names = list(enhancers)
        rows.append({"gene": gene, "term": "intercept", "coef": 6.0,
                     "se": 0.01, "pval": 0.0})
        for enh in names:
            true = sum(UNIT_EFFECT[f"{gene}_{enh}|{m}"] for m in enhancers[enh])
            rows.append({"gene": gene, "term": enh,
                         "coef": true + RNG.normal(0, 0.01),
                         "se": 0.01, "pval": 1e-6})
        for a, b in itertools.combinations(names, 2):
            rows.append({"gene": gene, "term": f"{a}:{b}",
                         "coef": -0.04 + RNG.normal(0, 0.005),
                         "se": 0.01, "pval": 1e-4})
    return pd.DataFrame(rows)


warnings.simplefilter("ignore")
print("building synthetic cross-enhancer run")
scram = build_scramble_frame()
print(f"  {len(scram)} rows, {scram['name'].nunique()} genes")

# --- the RNA arm, exactly as the handover documents it --------------------- #
res_rna = glm.fit_all_genes(scram, response_col="RNA_counts",
                            family="gamma", max_order=2)
beta_rna = tc.add_bh_correction(glm.collect_terms(res_rna))
inters_rna = glm.classify_interaction_terms(beta_rna)
pair_agg = glm.aggregate_pairs_to_enhancer(inters_rna)
print(f"  inters_rna {inters_rna.shape}, pair_agg {pair_agg.shape}")

# --- the enhancer level ---------------------------------------------------- #
enh_coef = tc.add_bh_correction(build_enhancer_level())
enh_mains, enh_inters = ea.split_coef_table(enh_coef)
print(f"  enh_mains {enh_mains.shape}, enh_inters {enh_inters.shape}")

# --- a synthetic screen ---------------------------------------------------- #
measured_pairs_raw = pd.DataFrame({
    "enhA": enh_inters["full_a"], "enhB": enh_inters["full_b"],
    "interaction_beta": enh_inters["coef"] * 2 + RNG.normal(0, 0.02,
                                                            len(enh_inters)),
    "source_file": "SIM", "": RNG.uniform(0, 1, len(enh_inters)),
})
measured_singles_raw = pd.DataFrame({
    "enhancer": enh_mains["full_a"],
    "beta": enh_mains["coef"] * 1.2 + RNG.normal(0, 0.02, len(enh_mains)),
    "source_dataset": "SIM",
})
measured_pairs = cl.normalise_measured_pairs(measured_pairs_raw)
measured_singles = cl.normalise_measured_singles(measured_singles_raw)

# --- G1 and G5 ------------------------------------------------------------- #
linked = cl.link_motif_pairs_to_enhancers(pair_agg, enh_inters, measured_pairs)
assert linked["enh_coef"].notna().all(), "enhancer coefficients failed to join"
assert linked["interaction_beta"].notna().all(), "measured pairs failed to join"
pair_link = cl.summarise_pair_link(linked)
within_gene = cl.summarise_within_gene_link(linked)
print(f"  G1 linked {linked.shape}, all keys matched")

# --- G2 -------------------------------------------------------------------- #
motifs = tc.add_bh_correction(
    tc.build_motif_effects(scram, value_col="RNA_counts", enhancer_col="name"))
# The within-enhancer run keys on the enhancer; the cross run keys on the gene,
# so the motif labels already carry the enhancer and must be split out here.
motifs["enhancer"] = (motifs["enhancer"] + "_"
                      + motifs["motif"].str.split("|").str[0])
motifs["motif"] = motifs["motif"].str.split("|").str[-1]
enh_agg = cl.aggregate_motifs_per_enhancer(motifs)
motif_linked = cl.link_motif_effects_to_enhancers(enh_agg, enh_mains,
                                                  measured_singles)
assert motif_linked["enh_coef"].notna().all(), "enhancer main effects failed to join"
motif_link = cl.summarise_motif_enhancer_link(motif_linked)
print(f"  G2 linked {motif_linked.shape}, all keys matched")
print(f"     motif_strongest vs in silico beta_i: "
      f"r={motif_link['insilico']['motif_strongest']['pearson_r']:+.3f}")

# --- G3 -------------------------------------------------------------------- #
seqlets = pd.DataFrame([
    {"name": gene, "example_idx": f"{enh}|{motif}",
     "start_seq": 20 + 40 * i, "end_seq": 30 + 40 * i}
    for gene, enhancers in GENES.items()
    for enh, motifs_ in enhancers.items()
    for i, motif in enumerate(motifs_)
])
pairs = tc.build_pairwise_interactions(scram, value_col="RNA_counts",
                                       enhancer_col="name", seqlets=seqlets)
spacing = cl.summarise_spacing_partial(pairs)
print(f"  G3 pairs {pairs.shape}, spacing partial r="
      f"{spacing['partial_r']:+.3f} on n={spacing['partial_n']}")

# --- rendering ------------------------------------------------------------- #
text = cl.format_cross_level_summary(pair_link, within_gene, motif_link, spacing)
print(f"\nrendered summary: {len(text.splitlines())} lines")
print("\n".join(text.splitlines()[:14]))
print("\nintegration test passed")
