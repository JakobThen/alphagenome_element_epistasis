"""End-to-end smoke test for the enhancer-level analysis and figures.

Builds coefficient tables with the same schema the notebook produces, with
known super-additive and sub-additive genes planted, then runs every plotting
function and assembles both figures.
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

# Example figures land next to the tests; the folder is git-ignored.
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import element_epistasis.enhancer_analysis as ea
import element_epistasis.enhancer_plots as ep
import element_epistasis.tf_cooperativity as tc

RNG = np.random.default_rng(11)


def simulate(n_genes=14):
    """Build GLM coefficient tables, comparison tables and enhancer coordinates."""
    coef_rows, atac_rows, coop_rows, epi_rows, bed_rows = [], [], [], [], []
    single_rows, pair_rows = [], []

    position = 1_000_000
    for g in range(n_genes):
        gene = f"GENE{g:02d}"
        n_enh = int(RNG.integers(2, 5))
        enhancers = [f"enh{i + 1}" for i in range(n_enh)]

        mains = {e: float(RNG.uniform(-0.9, -0.1)) for e in enhancers}
        regime = g % 3
        coupling = {0: -0.35, 1: 0.35, 2: 0.0}[regime]
        baseline = float(RNG.uniform(5, 60))

        coef_rows.append(dict(
            gene=gene, term="intercept", coef=float(np.log(baseline)),
            se=0.05, ci_lo=0.0, ci_hi=0.0, pval=0.0, family="gamma",
        ))

        for e in enhancers:
            start = position
            end = start + int(RNG.integers(300, 900))
            position = end + int(RNG.integers(2_000, 90_000))
            bed_rows.append(dict(chrom="chr9", start=start, end=end,
                                 name=f"{gene}_{e}"))

            se = abs(mains[e]) * RNG.uniform(0.08, 0.2)
            coef_rows.append(dict(
                gene=gene, term=e, coef=mains[e], se=se,
                ci_lo=mains[e] - 1.96 * se, ci_hi=mains[e] + 1.96 * se,
                pval=float(2 * (1 - min(0.999999, abs(mains[e]) / se / 4))),
                family="gamma", baseline=baseline,
            ))
            atac_rows.append(dict(
                gene=gene, term=e, coef=mains[e] * RNG.uniform(0.7, 1.3),
                se=se, pval=RNG.uniform(0, 0.2), family="gamma",
            ))
            single_rows.append(dict(
                enhancer=f"{gene}_{e}", gene=gene, coef=mains[e],
                interaction_beta=mains[e] * RNG.uniform(0.5, 1.5)
                + RNG.normal(0, 0.08),
                measured_pval=float(RNG.uniform(0, 0.3)),
            ))

        for a, b in itertools.combinations(enhancers, 2):
            value = coupling + RNG.normal(0, 0.06)
            se = abs(value) * RNG.uniform(0.1, 0.3) + 0.02
            coef_rows.append(dict(
                gene=gene, term=f"{a}:{b}", coef=value, se=se,
                ci_lo=value - 1.96 * se, ci_hi=value + 1.96 * se,
                pval=float(2 * (1 - min(0.999999, abs(value) / se / 4))),
                family="gamma", baseline=baseline,
                main_a=mains[a], main_b=mains[b],
            ))
            atac_rows.append(dict(
                gene=gene, term=f"{a}:{b}",
                coef=value * RNG.uniform(0.6, 1.4) + RNG.normal(0, 0.05),
                se=se, pval=float(RNG.uniform(0, 0.2)), family="gamma",
            ))
            pair_rows.append(dict(
                enhA=f"{gene}_{a}", enhB=f"{gene}_{b}", gene=gene, coef=value,
                interaction_beta=value * RNG.uniform(0.4, 1.6)
                + RNG.normal(0, 0.07),
                measured_pval=float(RNG.uniform(0, 0.3)),
            ))
            epi_rows.append(dict(enhancer=gene, tfA=f"{gene}_{a}",
                                 tfB=f"{gene}_{b}", interaction=value))

        summed = sum(mains.values())
        observed = summed + coupling * (n_enh - 1)
        coop_rows.append(dict(name=gene, n_seqlets=n_enh, n_singles=n_enh,
                              additive_prediction=summed, observed=observed,
                              cooperativity=observed - summed,
                              sum_d_rna=summed, d_rna_all=observed))

    scram_rows = []
    for gene, group in pd.DataFrame(bed_rows).groupby(
            pd.DataFrame(bed_rows).name.str.split("_").str[0]):
        enhancers = list(group.name)
        mains = {e: float(RNG.uniform(-0.9, -0.1)) for e in enhancers}
        coupling = {0: -0.35, 1: 0.35, 2: 0.0}[int(gene[4:6]) % 3]
        base = float(RNG.uniform(5, 60))
        for order in range(len(enhancers) + 1):
            for combo in itertools.combinations(enhancers, order):
                effect = sum(mains[e] for e in combo)
                if order >= 2:
                    effect += coupling * (order - 1)
                value = base * np.exp(effect)
                scram_rows.append(dict(
                    name=gene, n_scrambled=order, scrambled_seqlets=list(combo),
                    target_enhancer=enhancers, RNA_counts=value,
                    RNA_counts_sd=value * 0.04,
                ))

    return (pd.DataFrame(coef_rows), pd.DataFrame(atac_rows),
            pd.DataFrame(coop_rows), pd.DataFrame(epi_rows),
            pd.DataFrame(bed_rows), pd.DataFrame(single_rows),
            pd.DataFrame(pair_rows), pd.DataFrame(scram_rows))


def main():
    coef, atac, coop, epi, bed, single_comp, pair_comp, scram_agg = simulate()
    print(f"simulated {coef.gene.nunique()} genes, {len(coef)} coefficients, "
          f"{len(bed)} enhancers")

    mains, inters = ea.split_coef_table(coef)
    inters = tc.add_bh_correction(inters)
    mains = tc.add_bh_correction(mains)
    # attach_main_effects takes the UNSPLIT table, which still has the intercept.
    inters = ea.attach_main_effects(inters, coef)
    inters = ea.annotate_pair_distance(inters, bed)

    # The real ATAC table arrives already reshaped, so exercise both paths.
    _, atac_inters = ea.split_coef_table(atac)
    atac_inters = tc.add_bh_correction(atac_inters)
    reshaped = atac_inters[["gene", "enhA", "enhB", "coef", "pval"]] \
        if {"enhA", "enhB"}.issubset(atac_inters.columns) else \
        atac_inters.rename(columns={"full_a": "enhA", "full_b": "enhB"})[
            ["gene", "enhA", "enhB", "coef", "pval"]]
    assert len(ea.normalise_interaction_table(reshaped)) == len(atac_inters)

    print(f"main effects {len(mains)}, interactions {len(inters)}, "
          f"{int(inters.significant.sum())} significant at FDR < 0.05")
    print(f"distance annotated: {inters.distance_bp.notna().sum()} of {len(inters)}")

    summaries = {
        "single enhancers": ea.summarise_validation(
            single_comp, insilico_p_col="pval"),
        "enhancer pairs": ea.summarise_validation(
            pair_comp, insilico_p_col="pval"),
    }
    for k, v in summaries.items():
        print(f"  {k}: n={v['n']}, concordance={v['concordance']:.0%}, "
              f"in silico significant={v['n_insilico']}")

    comparison = pd.DataFrame([
        dict(setting=s, gene=f"GENE{i:02d}", converged=True,
             aic=float(RNG.uniform(*rng_range)), deviance=np.nan, n_obs=100,
             mean_se=float(RNG.uniform(*se_range)))
        for s, rng_range, se_range in [("gamma, order 2", (200, 260), (0.05, 0.12)),
                                       ("gamma, order 3", (205, 275), (0.10, 0.25)),
                                       ("nb, order 2", (260, 340), (0.08, 0.20))]
        for i in range(14)
    ])

    # Planted regimes must come back with the right sign.
    inters["regime"] = [int(g[4:6]) % 3 for g in inters.gene]
    recovered = inters.groupby("regime").coef.mean()
    print("\nplanted regime -> recovered mean interaction")
    for regime, label in [(0, "super-additive (expect < 0)"),
                          (1, "sub-additive   (expect > 0)"),
                          (2, "additive       (expect ~ 0)")]:
        print(f"  {label}: {recovered[regime]:+.3f}")
    assert recovered[0] < -0.1 and recovered[1] > 0.1 and abs(recovered[2]) < 0.1

    fig = ep.make_main_figure(inters, coop, single_comp, pair_comp, summaries)
    fig.savefig(OUT_DIR / "example_enhancer_main.png", dpi=110,
                bbox_inches="tight")
    plt.close(fig)

    fig = ep.make_supplementary_figure(inters, inters, atac_inters,
                                       comparison=comparison)
    fig.savefig(OUT_DIR / "example_enhancer_supp.png", dpi=110,
                bbox_inches="tight")
    plt.close(fig)

    # The epistasis map now stands alone at its own size.
    ax = ep.plot_epistasis_heatmap(epi)
    ax.get_figure().savefig(OUT_DIR / "example_epistasis_map.png",
                            dpi=110, bbox_inches="tight")
    plt.close(ax.get_figure())

    # Worked example for one gene.
    gene = "GENE00"
    rna_result = dict(
        gene=gene, family="gamma", max_order=2, n_obs=120, converged=True,
        coef_table=coef[coef.gene == gene].assign(
            qval=coef[coef.gene == gene].pval),
    )
    fig = ep.make_gene_figure(gene, rna_result, epi, scram_agg)
    fig.savefig(OUT_DIR / "example_gene_figure.png", dpi=110,
                bbox_inches="tight")
    plt.close(fig)

    # Panels not used in either assembled figure still have to work.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ep.plot_coef_forest(
        dict(gene="GENE00", family="gamma", max_order=2, n_obs=100,
             converged=True,
             coef_table=coef[coef.gene == "GENE00"].assign(
                 qval=coef[coef.gene == "GENE00"].pval)),
        ax=axes[0])
    tf_coop = pd.DataFrame(dict(
        enhancer=mains.full_a, cooperativity=RNG.normal(0, 0.5, len(mains))))
    ep.plot_cross_level(tf_coop, mains, ax=axes[1])
    ep.plot_epistasis_heatmap(epi[epi.enhancer == "GENE00"], ax=axes[2])
    fig.tight_layout()
    fig.savefig(OUT_DIR / "example_enhancer_extra.png", dpi=110,
                bbox_inches="tight")
    plt.close(fig)

    print("\nAll enhancer-level figures rendered.")


if __name__ == "__main__":
    main()
