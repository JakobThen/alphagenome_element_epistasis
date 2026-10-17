"""Correctness checks for tf_cooperativity.

The interaction contrast is tested against synthetic data where the true answer
is known by construction, which is the only way to be sure the sign is right.
"""

import numpy as np
import pandas as pd

import element_epistasis.tf_cooperativity as tc


def make_scramble_frame(y0, ya, yb, yab, n_shuffles=10, seed=0, name="ENH"):
    """Build a minimal scramble frame with two motifs and known readouts."""
    rng = np.random.default_rng(seed)
    rows = [dict(name=name, n_scrambled=0, scrambled_seqlets=[],
                 shuffle_idx=np.nan, ATAC_counts=y0)]

    for label, value in [(["A"], ya), (["B"], yb), (["A", "B"], yab)]:
        for i in range(n_shuffles):
            jitter = 1.0 + rng.normal(0, 0.01)
            rows.append(dict(name=name, n_scrambled=len(label),
                             scrambled_seqlets=label, shuffle_idx=i,
                             ATAC_counts=value * jitter))
    return pd.DataFrame(rows)


def test_multiplicative_independence_gives_zero():
    """Two motifs each halving accessibility independently must give I = 0."""
    y0 = 100.0
    df = make_scramble_frame(y0=y0, ya=50.0, yb=50.0, yab=25.0)
    res = tc.build_pairwise_interactions(df, scale="log")

    assert len(res) == 1
    assert abs(res.interaction.iloc[0]) < 0.02, res.interaction.iloc[0]
    print(f"  independence     I = {res.interaction.iloc[0]:+.4f}  (expect ~0)")


def test_synergy_is_negative():
    """Losing more together than predicted must give a negative interaction."""
    df = make_scramble_frame(y0=100.0, ya=50.0, yb=50.0, yab=10.0)
    res = tc.build_pairwise_interactions(df, scale="log")

    value = res.interaction.iloc[0]
    assert value < 0, value
    # log(10) - log(50) - log(50) + log(100) = log(0.4)
    assert abs(value - np.log(0.4)) < 0.02
    print(f"  synergy          I = {value:+.4f}  (expect {np.log(0.4):+.4f})")


def test_redundancy_is_positive():
    """A buffered pair that loses less than predicted must give positive I."""
    df = make_scramble_frame(y0=100.0, ya=50.0, yb=50.0, yab=45.0)
    res = tc.build_pairwise_interactions(df, scale="log")

    value = res.interaction.iloc[0]
    assert value > 0, value
    assert abs(value - np.log(1.8)) < 0.02
    print(f"  redundancy       I = {value:+.4f}  (expect {np.log(1.8):+.4f})")


def test_old_formula_fails_the_null():
    """Demonstrate the original notebook formula on the independent case."""
    y0, ya, yb, yab = 100.0, 50.0, 50.0, 25.0
    d_a, d_b, d_ab = ya - y0, yb - y0, yab - y0

    correct = d_ab - d_a - d_b
    notebook = (y0 - yab) - d_a - d_b

    print(f"  additive (linear) truth        I = {correct:+.1f}")
    print(f"  notebook formula on same data  I = {notebook:+.1f}")
    assert notebook != correct
    # The notebook value is the negated sum of all three effects.
    assert np.isclose(notebook, -(d_ab + d_a + d_b))


def test_cooperativity_sign_matches_pairwise():
    """The two-motif cooperativity score must equal the pairwise interaction."""
    df = make_scramble_frame(y0=100.0, ya=50.0, yb=50.0, yab=10.0)
    pair = tc.build_pairwise_interactions(df, scale="log").interaction.iloc[0]
    coop = tc.build_cooperativity_table(df, scale="log").cooperativity.iloc[0]

    assert abs(pair - coop) < 0.02, (pair, coop)
    print(f"  pairwise {pair:+.4f} vs cooperativity {coop:+.4f}  (must agree)")


def test_unique_labels_preserve_design():
    """Repeated motif names must not collapse distinct combinations."""
    seqlets = pd.DataFrame(dict(
        name=["E1", "E1", "E1", "E2"],
        example_idx=["FOS", "FOS", "GATA2", "MYB"],
        start=[10, 100, 200, 50],
        end=[20, 110, 210, 60],
        attribution=[5.0, 4.0, 3.0, 2.0],
    ))
    out = tc.make_seqlet_labels_unique(seqlets)

    e1 = out[out.name == "E1"].example_idx.tolist()
    assert len(set(e1)) == 3, e1
    assert "GATA2" in e1 and "MYB" in out[out.name == "E2"].example_idx.tolist()
    print(f"  E1 labels -> {e1}")


def test_cap_drops_overlaps_and_weak_seqlets():
    """Capping must keep the strongest non-overlapping seqlets."""
    seqlets = pd.DataFrame(dict(
        name=["E1"] * 6,
        example_idx=list("ABCDEF"),
        start=[0, 5, 50, 100, 150, 200],
        end=[10, 15, 60, 110, 160, 210],   # A and B overlap
        attribution=[9.0, 8.0, 7.0, 6.0, 5.0, 4.0],
    ))
    out = tc.cap_seqlets_per_enhancer(seqlets, max_seqlets=3)

    kept = out.example_idx.tolist()
    assert "B" not in kept, kept          # dropped for overlapping stronger A
    assert kept == ["A", "C", "D"], kept
    print(f"  kept -> {kept}")


def test_bh_correction_matches_reference():
    """BH output must match a hand-computed reference."""
    df = pd.DataFrame(dict(pval=[0.001, 0.008, 0.039, 0.041, 0.042, 0.06]))
    out = tc.add_bh_correction(df)

    expected = np.array([0.006, 0.024, 0.0504, 0.0504, 0.0504, 0.06])
    assert np.allclose(out.qval.values, expected, atol=1e-6), out.qval.values
    assert out.significant.tolist() == [True, True, False, False, False, False]
    print(f"  qvals -> {np.round(out.qval.values, 4)}")


def test_design_budget_shrinks():
    """The reduced design must cost less than the full power set."""
    seqlets = pd.DataFrame(dict(
        name=["E1"] * 5 + ["E2"] * 3,
        example_idx=list("ABCDEXYZ"),
    ))
    full = tc.summarise_design_budget(seqlets, design="full")
    reduced = tc.summarise_design_budget(seqlets, design="reduced")

    f = int(full[full.name == "TOTAL"].n_sequences.iloc[0])
    r = int(reduced[reduced.name == "TOTAL"].n_sequences.iloc[0])
    assert r < f
    print(f"  full {f} sequences vs reduced {r} ({100 * (1 - r / f):.0f}% saving)")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        print(f"\n{test.__name__}")
        test()
    print(f"\nAll {len(tests)} tests passed.")
