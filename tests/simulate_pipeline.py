"""End-to-end smoke test on simulated data.

Generates a scramble frame with the same schema the notebook produces, with
known synergy and redundancy planted in specific enhancers, then runs the full
analysis and plotting pipeline. Confirms that the tables and figures build and
that the planted structure is recovered with the correct sign.
"""

import matplotlib
matplotlib.use("Agg")

from pathlib import Path

# Example figures land next to the tests; the folder is git-ignored.
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import element_epistasis.tf_cooperativity as tc
import element_epistasis.tf_plots as tp

RNG = np.random.default_rng(7)
FACTORS = ["GATA1", "TAL1", "SPI1", "KLF1", "NFE2", "CTCF", "MYB", "RUNX1"]
N_SHUFFLES = 10


LENGTH = 313


def simulate(n_enhancers=28):
    """Build a scramble frame plus its seqlet table with planted interactions."""
    scram_rows, seqlet_rows, control_rows = [], [], []
    attributions = RNG.uniform(0, 0.4, size=(n_enhancers, LENGTH))

    for e in range(n_enhancers):
        name = f"GENE{e:02d}_enh1"
        n_motifs = int(RNG.integers(2, 6))
        motifs = list(RNG.choice(FACTORS, size=n_motifs, replace=False))

        # Motif positions along a 313 bp enhancer, non-overlapping by construction.
        starts = np.sort(RNG.choice(np.arange(5, 290, 12), size=n_motifs,
                                    replace=False))
        widths = RNG.integers(6, 13, size=n_motifs)

        base = float(RNG.uniform(20, 200))
        # Each motif contributes a multiplicative loss when scrambled.
        effects = {m: float(RNG.uniform(-1.1, -0.15)) for m in motifs}

        # Planted interaction: a third of enhancers synergise, a third buffer.
        regime = e % 3
        coupling = {0: -0.45, 1: 0.45, 2: 0.0}[regime]

        for m, s, w, in zip(motifs, starts, widths):
            seqlet_rows.append(dict(
                name=name, seq_idx=e, example_idx=m,
                start_seq=int(s), end_seq=int(s + w),
                attribution=float(-effects[m] * RNG.uniform(3, 6)),
            ))
            attributions[e, s:s + w] = -effects[m] * RNG.uniform(3, 6)

        # Control windows: matched length, near-zero true effect.
        free = [p for p in range(5, LENGTH - 15)
                if all(not (p < s + w + 3 and s - 3 < p + 8)
                       for s, w in zip(starts, widths))]
        for k, start in enumerate(RNG.choice(free, size=3, replace=False), start=1):
            control_rows.append(dict(
                name=name, seq_idx=e, example_idx=f"ctrl#{k}",
                start_seq=int(start), end_seq=int(start + 8),
                attribution=float(attributions[e, start:start + 8].mean()),
                is_control=True,
            ))
            for shuffle_idx in range(N_SHUFFLES):
                # A generic window carries a small, mostly technical effect.
                noise = RNG.normal(-0.03, 0.05)
                scram_rows.append(dict(
                    name=name, n_scrambled=1,
                    scrambled_seqlets=[f"ctrl#{k}"], unscrambled_seqlets=[],
                    shuffle_idx=shuffle_idx,
                    ATAC_counts=float(base * np.exp(noise)),
                    RNA_counts=[2.0], _run="control",
                ))
        control_rows.append(None)  # placeholder removed below

        combos = tc.enumerate_combinations(n_motifs, design="reduced")
        for combo in combos:
            labels = [motifs[i] for i in combo]
            log_effect = sum(effects[m] for m in labels)
            # Interaction only applies once two or more motifs are removed.
            if len(combo) >= 2:
                log_effect += coupling * (len(combo) - 1)

            n_reps = 1 if not combo else N_SHUFFLES
            for shuffle_idx in range(n_reps):
                noise = RNG.normal(0, 0.04) if combo else 0.0
                atac = base * np.exp(log_effect + noise)
                # RNA tracks ATAC with attenuation and extra noise.
                rna = 2.0 * np.exp(0.6 * log_effect + RNG.normal(0, 0.05))
                scram_rows.append(dict(
                    name=name, _run="main",
                    n_scrambled=len(combo),
                    scrambled_seqlets=labels,
                    unscrambled_seqlets=[m for m in motifs if m not in labels],
                    shuffle_idx=shuffle_idx if combo else np.nan,
                    ATAC_counts=float(atac),
                    RNA_counts=[float(rna)],
                ))

    controls = pd.DataFrame([r for r in control_rows if r is not None])
    frame = pd.DataFrame(scram_rows)
    main = frame[frame["_run"] == "main"].drop(columns="_run")
    # The control run needs its own baseline rows, copied from the main run.
    baseline = main[main.n_scrambled == 0]
    control_scram = pd.concat([baseline, frame[frame["_run"] == "control"]
                               .drop(columns="_run")], ignore_index=True)
    return main, pd.DataFrame(seqlet_rows), control_scram, controls


def main():
    scram, seqlets, control_scram, control_windows = simulate()
    print(f"simulated {scram.name.nunique()} enhancers, {len(scram)} sequences, "
          f"{len(seqlets)} seqlets, {len(control_windows)} control windows")

    controls = tc.build_motif_effects(control_scram, value_col="ATAC_counts",
                                      seqlets=control_windows)

    motifs = tc.build_motif_effects(scram, value_col="ATAC_counts", seqlets=seqlets)
    motifs = tc.add_bh_correction(motifs)

    pairs = tc.build_pairwise_interactions(scram, value_col="ATAC_counts",
                                           seqlets=seqlets)
    pairs = tc.add_bh_correction(pairs)

    coop_atac = tc.add_bh_correction(
        tc.build_cooperativity_table(scram, value_col="ATAC_counts"))
    coop_rna = tc.add_bh_correction(
        tc.build_cooperativity_table(scram, value_col="RNA_counts"))

    print(f"\nmotif effects : {len(motifs):>4} rows, "
          f"{int(motifs.significant.sum())} significant")
    print(f"pairs         : {len(pairs):>4} rows, "
          f"{int(pairs.significant.sum())} significant")
    print(f"coop ATAC     : {len(coop_atac):>4} rows, "
          f"{int(coop_atac.significant.sum())} significant")
    print(f"spacing column present: {'spacing_bp' in pairs.columns}")

    # The planted regimes must come back with the right sign.
    coop_atac["regime"] = [int(n[4:6]) % 3 for n in coop_atac.enhancer]
    summary = coop_atac.groupby("regime").cooperativity.mean()
    print("\nplanted regime -> recovered mean cooperativity")
    for regime, label in [(0, "synergy   (expect < 0)"),
                          (1, "redundancy(expect > 0)"),
                          (2, "additive  (expect ~ 0)")]:
        print(f"  {label}: {summary[regime]:+.3f}")
    assert summary[0] < -0.1, summary[0]
    assert summary[1] > 0.1, summary[1]
    assert abs(summary[2]) < 0.1, summary[2]

    print(f"control effects: {len(controls)} rows, "
          f"median delta {controls.delta.median():+.3f} "
          f"(motifs: {motifs.delta.median():+.3f})")

    fig = tp.make_main_figure(coop_atac, coop_rna, pairs, motifs, controls=controls)
    fig.savefig(OUT_DIR / "example_main_figure.png", dpi=110,
                bbox_inches="tight")
    plt.close(fig)

    split = tc.build_saturation_split(scram, value_col="ATAC_counts")
    floor_df = tc.build_floor_check(scram, value_col="ATAC_counts")
    dose = tc.build_dose_response(scram, value_col="ATAC_counts", seqlets=seqlets)
    print(f"split-shuffle pairs: {len(split)} | "
          f"pairs with null below floor: {int(floor_df.null_below_floor.sum())}"
          f"/{len(floor_df)}")

    fig = tp.make_supplementary_figure(motifs, coop_atac, pairs,
                                       controls=controls, split=split,
                                       floor_df=floor_df, dose=dose)
    fig.savefig(OUT_DIR / "example_supplementary_figure.png", dpi=110,
                bbox_inches="tight")
    plt.close(fig)

    print("\nBoth figures rendered. Planted structure recovered with correct sign.")


if __name__ == "__main__":
    main()
