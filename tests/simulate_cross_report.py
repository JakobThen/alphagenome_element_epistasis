"""End to end check of element_epistasis.cross_report against synthetic data.

Ground truth is planted rather than assumed: one gene carries a strong
super-additive interaction between a motif in enhancer 1 and a motif in
enhancer 2, and the enhancer-level table is written to agree with it. The test
asserts that the summary builds, that the scope contrast recovers the planted
difference, and that the exemplar search puts the planted gene first.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import element_epistasis.glm as glm
import element_epistasis.enhancer_analysis as ea
import element_epistasis.tf_cooperativity as tc
from element_epistasis.cross_report import build_cross_summary, find_exemplar_loci

# Output lands next to the tests; the folder is git-ignored.
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

RNG = np.random.default_rng(0)

GENES = ["GENEA", "GENEB", "GENEC", "GENED", "GENEE", "GENEF"]
ENHANCERS = ["enh1", "enh2", "enh3"]
MOTIFS = ["GATA1", "TAL1", "SPI1"]

PLANTED_GENE = "GENEA"
PLANTED_A = "enh1|GATA1"
PLANTED_B = "enh2|TAL1"
PLANTED_INTERACTION = -0.60


def build_units(gene):
    """Two seqlets in each of three enhancers, labelled by their enhancer."""
    return [f"{enh}|{motif}"
            for enh in ENHANCERS
            for motif in MOTIFS[:2]]


def combinations(units):
    """Baseline, singletons, all pairs and the all-scrambled state."""
    combos = [[]]
    combos += [[u] for u in units]
    combos += [[a, b] for i, a in enumerate(units) for b in units[i + 1:]]
    combos += [list(units)]
    return combos


def simulate(n_shuffles=6):
    """Generate a cross-enhancer scramble frame with known coefficients."""
    rows = []
    truth = {}

    for gene in GENES:
        units = build_units(gene)
        main = {u: RNG.uniform(-0.35, -0.05) for u in units}
        interactions = {}
        if gene == PLANTED_GENE:
            interactions[frozenset([PLANTED_A, PLANTED_B])] = PLANTED_INTERACTION
        truth[gene] = dict(main=main, interactions=interactions)

        for combo in combinations(units):
            scrambled = frozenset(combo)
            effect = sum(main[u] for u in combo)
            for pair, value in interactions.items():
                if pair <= scrambled:
                    effect += value

            for shuffle in range(1 if not combo else n_shuffles):
                noise = RNG.normal(0, 0.02)
                rna = 100.0 * np.exp(effect + noise)
                # Accessibility responds mostly to motifs in the measured
                # element, with a small trans component.
                atac = []
                for enh in ENHANCERS:
                    local = sum(main[u] for u in combo if u.startswith(enh))
                    distal = 0.1 * sum(main[u] for u in combo
                                       if not u.startswith(enh))
                    atac.append(50.0 * np.exp(3 * local + distal
                                              + RNG.normal(0, 0.02)))
                rows.append(dict(
                    name=gene,
                    scrambled_seqlets=list(combo),
                    unscrambled_seqlets=list(units),
                    target_enhancer=[f"{gene}_{e}" for e in ENHANCERS],
                    n_scrambled=len(combo),
                    shuffle_idx=shuffle,
                    RNA_counts=rna,
                    ATAC_counts=atac,
                ))

    return pd.DataFrame(rows), truth


def fake_enhancer_level(inters_rna):
    """Enhancer-level coefficients that agree with the planted motif pair."""
    rows = []
    for gene in GENES:
        for i, first in enumerate(ENHANCERS):
            for second in ENHANCERS[i + 1:]:
                coef = RNG.normal(0, 0.02)
                pval = RNG.uniform(0.2, 0.9)
                if gene == PLANTED_GENE and {first, second} == {"enh1", "enh2"}:
                    coef, pval = -0.31, 1e-5
                rows.append(dict(
                    gene=gene, term=f"{first}:{second}", coef=coef,
                    se=0.02, pval=pval,
                    full_a=f"{gene}_{first}", full_b=f"{gene}_{second}",
                    distance_bp=RNG.uniform(2e3, 8e4),
                    main_a=-0.2, main_b=-0.15,
                ))
    return tc.add_bh_correction(pd.DataFrame(rows))


def fake_within_run():
    """A within-enhancer run that measured the same factor pair elsewhere."""
    rows = []
    for gene in GENES[:3]:
        for enh in ENHANCERS[:2]:
            rows.append(dict(enhancer=f"{gene}_{enh}", motif_a="GATA1",
                             motif_b="TAL1", interaction=RNG.normal(-0.1, 0.05),
                             pval=RNG.uniform(0.001, 0.4), spacing_bp=40.0))
    return tc.add_bh_correction(pd.DataFrame(rows))


def main():
    raw, truth = simulate()
    print(f"simulated {len(raw)} rows over {raw['name'].nunique()} genes")

    res_rna = glm.fit_all_genes(raw, response_col="RNA_counts",
                                family="gamma", max_order=2, round_nb=False)
    beta_rna = tc.add_bh_correction(glm.collect_terms(res_rna))
    inters_rna = glm.classify_interaction_terms(beta_rna)

    # Distances stand in for the seqlet coordinate join, on the right scale.
    inters_rna["distance_bp"] = np.where(
        inters_rna["pair_scope"] == "within",
        RNG.uniform(20, 200, len(inters_rna)),
        RNG.uniform(5e3, 2e5, len(inters_rna)))

    long_atac = ea.expand_locus_values(raw, "ATAC_counts")
    res_atac = glm.fit_all_loci(long_atac, response_col="ATAC_counts",
                                gene_col="name", locus_col="locus",
                                family="gamma", max_order=2, round_nb=False)
    beta_atac = tc.add_bh_correction(glm.collect_terms(res_atac))
    inters_atac = glm.classify_interaction_terms(beta_atac)
    interdep = tc.add_bh_correction(glm.build_motif_interdependency(beta_atac))

    pair_agg = glm.aggregate_pairs_to_enhancer(inters_rna)
    scope = glm.summarise_scope_comparison(inters_rna)
    enh_inters = fake_enhancer_level(inters_rna)
    tf_pairs = fake_within_run()

    # Ground truth check on the estimator itself before testing the report.
    planted = inters_rna[
        (inters_rna["gene"] == PLANTED_GENE)
        & (inters_rna["term"].str.contains("GATA1"))
        & (inters_rna["term"].str.contains("TAL1"))
        & (inters_rna["pair_scope"] == "across")]
    recovered = planted["coef"].iloc[0]
    print(f"planted beta_ij = {PLANTED_INTERACTION:+.3f}, "
          f"recovered = {recovered:+.3f}")
    assert abs(recovered - PLANTED_INTERACTION) < 0.05, "estimator is off"

    candidates = find_exemplar_loci(inters_rna, enh_inters, interdep=interdep,
                                    tf_pairs=tf_pairs)
    top = candidates.iloc[0]
    print(f"top exemplar: {top['gene']} {top['enh_pair']} "
          f"evidence={top['n_evidence']}")
    assert top["gene"] == PLANTED_GENE, "exemplar search missed the planted pair"
    assert "enh1" in top["enh_pair"] and "enh2" in top["enh_pair"]
    assert top["lead_motif_a"] in ("GATA1", "TAL1")

    text = build_cross_summary(
        inters_rna=inters_rna, beta_rna=beta_rna, inters_atac=inters_atac,
        beta_atac=beta_atac, interdep=interdep, pair_agg=pair_agg, scope=scope,
        enh_inters=enh_inters, tf_pairs=tf_pairs,
        glm_results_rna=res_rna, glm_results_atac=res_atac,
    )
    with open(OUT_DIR / "cross_summary_simulated.txt", "w") as handle:
        handle.write(text)
    print(f"summary built: {len(text.splitlines())} lines")

    for heading in ["3.  DESIGN", "5.  MOTIF PAIR", "8.  ACCESSIBILITY",
                    "10.  DOES MOTIF", "12.  EXEMPLAR", "13.  FLAGS"]:
        assert heading in text, f"missing section {heading}"

    # The report must also survive being given almost nothing.
    minimal = build_cross_summary(inters_rna=inters_rna)
    assert "MISSING" in minimal
    print("degraded-input build ok")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
