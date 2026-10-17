"""Within-enhancer TF motif analysis: seqlet preparation, design, prediction driver and effect tables.

Effects are log fold changes against the unperturbed enhancer; the interaction
is I = delta_AB - delta_A - delta_B (negative: super-additive) and equals the
log-link GLM interaction coefficient.
"""

from __future__ import annotations

import itertools
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

_IDX_TO_NUC = np.array(["A", "C", "G", "T"])


# --------------------------------------------------------------------------- #
# 1. Reference sequence guard
# --------------------------------------------------------------------------- #

def reference_onehot_from_ism(ism_matrix: np.ndarray) -> np.ndarray:
    """Recover the reference one-hot sequence from an ISM attribution matrix.

    ISM scores sit at the reference base only, so argmax(|ISM|) is the reference.

    Args:
        ism_matrix: Attributions of shape (length, 4), ACGT order.

    Returns:
        One-hot array of shape (4, length).

    Raises:
        ValueError: If a position has no non-zero entry.
    """
    ism_matrix = np.asarray(ism_matrix)
    if (ism_matrix == 0).all(axis=1).any():
        raise ValueError(
            "ISM matrix contains all-zero positions; the reference base cannot "
            "be recovered by argmax and motif annotation would be corrupted."
        )

    indices = np.argmax(np.abs(ism_matrix), axis=1)
    onehot = np.zeros((4, len(indices)), dtype=np.float32)
    onehot[indices, np.arange(len(indices))] = 1.0
    return onehot


def check_ism_reference(
    ism_dict: dict,
    enhancers: pd.DataFrame,
    name_col: str = "name",
    seq_col: str = "seq",
    track: str = "ATAC",
) -> pd.DataFrame:
    """Compare ISM-derived reference sequences with the genome sequence.

    Args:
        ism_dict: Mapping of enhancer to track label to attribution array.
        enhancers: Table with the genomic sequence of each enhancer.
        name_col: Enhancer name column.
        seq_col: Sequence column.
        track: Track to reconstruct from.

    Returns:
        One row per enhancer with both sequences and the number of mismatches.
    """
    truth = dict(zip(enhancers[name_col], enhancers[seq_col].str.upper()))
    rows = []

    for name, tracks in ism_dict.items():
        onehot = reference_onehot_from_ism(tracks[track])
        recon = "".join(_IDX_TO_NUC[onehot.argmax(axis=0)])
        genomic = truth.get(name)

        if genomic is None:
            n_mismatch = np.nan
        else:
            n_mismatch = sum(a != b for a, b in zip(recon, genomic))

        rows.append(
            dict(name=name, reconstructed=recon, genomic=genomic,
                 n_mismatch=n_mismatch)
        )

    out = pd.DataFrame(rows)
    n_bad = int((out["n_mismatch"].fillna(-1) != 0).sum())
    if n_bad:
        warnings.warn(f"{n_bad} enhancer(s) failed the reference check.")
    return out


# --------------------------------------------------------------------------- #
# 2. Seqlet preparation
# --------------------------------------------------------------------------- #

def make_seqlet_labels_unique(
    seqlets: pd.DataFrame,
    enhancer_col: str = "name",
    label_col: str = "example_idx",
    start_col: str = "start",
) -> pd.DataFrame:
    """Suffix repeated motif labels within an enhancer ("FOS" -> "FOS#1", "FOS#2").

    Args:
        seqlets: One row per seqlet.
        enhancer_col: Parent enhancer column.
        label_col: Motif label column.
        start_col: Column ordering repeated occurrences.

    Returns:
        Copy of seqlets with labels unique within each enhancer.
    """
    out = seqlets.copy()

    def _relabel(group: pd.DataFrame) -> pd.Series:
        counts = group[label_col].value_counts()
        repeated = set(counts[counts > 1].index)
        if not repeated:
            return group[label_col]

        ordered = group.sort_values(start_col)
        occurrence = ordered.groupby(label_col).cumcount() + 1
        labels = ordered[label_col].where(
            ~ordered[label_col].isin(repeated),
            ordered[label_col] + "#" + occurrence.astype(str),
        )
        return labels.reindex(group.index)

    out[label_col] = (
        out.groupby(enhancer_col, group_keys=False).apply(_relabel)
    )
    return out


def cap_seqlets_per_enhancer(
    seqlets: pd.DataFrame,
    max_seqlets: int = 5,
    enhancer_col: str = "name",
    score_col: str = "attribution",
    start_col: str = "start",
    end_col: str = "end",
    enforce_non_overlap: bool = True,
) -> pd.DataFrame:
    """Keep the highest-scoring, non-overlapping seqlets of each enhancer.

    Args:
        seqlets: One row per seqlet.
        max_seqlets: Maximum seqlets kept per enhancer.
        enhancer_col: Parent enhancer column.
        score_col: Ranking column, descending.
        start_col: Seqlet start (0-based).
        end_col: Seqlet end (exclusive).
        enforce_non_overlap: Greedily drop seqlets overlapping a kept one.

    Returns:
        Filtered copy sorted by enhancer and start.
    """
    kept = []

    for _, group in seqlets.groupby(enhancer_col, sort=False):
        ranked = group.sort_values(score_col, ascending=False)
        chosen = []

        for _, row in ranked.iterrows():
            if len(chosen) >= max_seqlets:
                break
            if enforce_non_overlap and any(
                row[start_col] < c[end_col] and c[start_col] < row[end_col]
                for c in chosen
            ):
                continue
            chosen.append(row)

        kept.extend(chosen)

    out = pd.DataFrame(kept).sort_values([enhancer_col, start_col])
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 3. Design enumeration and budget
# --------------------------------------------------------------------------- #

def enumerate_combinations(n_seqlets: int, design: str = "reduced") -> list[tuple]:
    """List the seqlet index subsets to scramble.

    Args:
        n_seqlets: Number of seqlets in the enhancer.
        design: "full" (power set), "reduced" (baseline, singles, pairs and
            all-scrambled) or "singles" (baseline and singles).

    Returns:
        Sorted index tuples, starting with the empty tuple.
    """
    indices = range(n_seqlets)

    if design == "full":
        combos = [
            combo
            for order in range(n_seqlets + 1)
            for combo in itertools.combinations(indices, order)
        ]
    elif design == "reduced":
        combos = [()]
        combos += list(itertools.combinations(indices, 1))
        combos += list(itertools.combinations(indices, 2))
        if n_seqlets > 2:
            combos.append(tuple(indices))
    elif design == "singles":
        combos = [()] + list(itertools.combinations(indices, 1))
    else:
        raise ValueError(
            f"Unknown design '{design}'; use 'full', 'reduced' or 'singles'."
        )

    return sorted(set(combos), key=lambda c: (len(c), c))


def summarise_design_budget(
    seqlets: pd.DataFrame,
    n_shuffles: int = 10,
    design: str = "reduced",
    enhancer_col: str = "name",
) -> pd.DataFrame:
    """Count the sequences a design requires before predicting.

    Args:
        seqlets: One row per seqlet, after capping.
        n_shuffles: Shuffles per non-baseline combination.
        design: Design passed to enumerate_combinations.
        enhancer_col: Parent enhancer column.

    Returns:
        Sequence counts per enhancer with a TOTAL row.
    """
    rows = []
    for name, group in seqlets.groupby(enhancer_col, sort=True):
        n = len(group)
        n_combos = len(enumerate_combinations(n, design))
        # The unperturbed baseline is a single sequence; everything else is
        # replicated across shuffles.
        n_seqs = 1 + (n_combos - 1) * n_shuffles
        rows.append(dict(name=name, n_seqlets=n, n_combos=n_combos,
                         n_sequences=n_seqs))

    out = pd.DataFrame(rows)
    total = dict(name="TOTAL", n_seqlets=out.n_seqlets.sum(),
                 n_combos=out.n_combos.sum(), n_sequences=out.n_sequences.sum())
    return pd.concat([out, pd.DataFrame([total])], ignore_index=True)


def sample_control_windows(
    exclude_seqlets: pd.DataFrame,
    attributions: Any,
    enhancer_names: list[str],
    n_controls: int = 3,
    seq_idx_col: str = "seq_idx",
    start_col: str = "start_seq",
    end_col: str = "end_seq",
    attribution_percentile: float = 50.0,
    margin: int = 3,
    random_state: int = 0,
) -> pd.DataFrame:
    """Sample length-matched control windows outside called seqlets, for the specificity null.

    Args:
        exclude_seqlets: All called seqlets before capping; windows avoid them.
        attributions: Per-base attribution, shape (n_enhancers, length).
        enhancer_names: Enhancer names aligned with attributions.
        n_controls: Windows per enhancer.
        seq_idx_col: Column mapping a seqlet to its row in attributions.
        start_col: Seqlet start (0-based).
        end_col: Seqlet end (exclusive).
        attribution_percentile: Keep windows at or below this percentile of mean
            absolute attribution (100 samples uniformly).
        margin: Bases kept clear around each seqlet.
        random_state: Random seed.

    Returns:
        Control windows in the seqlet table layout, labelled "ctrl#1", ... with
        is_control set.
    """
    attributions = np.abs(np.asarray(attributions, dtype=float))
    rng = np.random.default_rng(random_state)

    widths_global = (exclude_seqlets[end_col] - exclude_seqlets[start_col])
    median_width = int(np.median(widths_global)) if len(widths_global) else 8

    rows = []
    by_index = exclude_seqlets.groupby(seq_idx_col)

    for seq_idx, name in enumerate(enhancer_names):
        track = attributions[seq_idx]
        length = len(track)

        # Positions already claimed by a called seqlet, plus a safety margin.
        blocked = np.zeros(length, dtype=bool)
        if seq_idx in by_index.groups:
            group = by_index.get_group(seq_idx)
            for _, row in group.iterrows():
                lo = max(0, int(row[start_col]) - margin)
                hi = min(length, int(row[end_col]) + margin)
                blocked[lo:hi] = True
            widths = (group[end_col] - group[start_col]).astype(int).tolist()
        else:
            widths = [median_width]

        # Enumerate every free window of every observed seqlet width.
        candidates = []
        for width in set(widths):
            width = int(width)
            if width <= 0 or width > length:
                continue
            for start in range(0, length - width + 1):
                if blocked[start:start + width].any():
                    continue
                candidates.append((start, start + width,
                                   float(track[start:start + width].mean())))

        if not candidates:
            warnings.warn(f"No free control window available for {name}.")
            continue

        candidate_df = pd.DataFrame(candidates,
                                    columns=["start", "end", "attribution"])
        cutoff = np.percentile(candidate_df["attribution"], attribution_percentile)
        eligible = candidate_df[candidate_df["attribution"] <= cutoff]
        if eligible.empty:
            eligible = candidate_df

        # Greedy non-overlapping draw so the controls are independent of one another.
        chosen, order = [], rng.permutation(len(eligible))
        for pos in order:
            row = eligible.iloc[pos]
            if len(chosen) >= n_controls:
                break
            if any(row["start"] < c["end"] and c["start"] < row["end"]
                   for c in chosen):
                continue
            chosen.append(row)

        for k, row in enumerate(chosen, start=1):
            rows.append({
                "name": name,
                seq_idx_col: seq_idx,
                "example_idx": f"ctrl#{k}",
                start_col: int(row["start"]),
                end_col: int(row["end"]),
                "attribution": float(row["attribution"]),
                "is_control": True,
            })

    out = pd.DataFrame(rows)
    if len(out) < n_controls * len(enhancer_names):
        warnings.warn(
            f"Drew {len(out)} control windows; "
            f"{n_controls * len(enhancer_names)} were requested."
        )
    return out


# --------------------------------------------------------------------------- #
# 4. Robust prediction driver
# --------------------------------------------------------------------------- #

def predict_with_backoff(
    model: Any,
    sequences: list[str],
    interval: Any,
    requested_outputs: list,
    ontology_terms: list[str],
    chunk_size: int = 64,
    max_workers: int = 6,
    min_workers: int = 1,
    max_retries: int = 6,
    base_wait: float = 15.0,
    verbose: bool = True,
) -> list:
    """Request AlphaGenome predictions in chunks, retrying failed chunks with fewer workers.

    Args:
        model: AlphaGenome dna_client model.
        sequences: Input sequences of equal length.
        interval: Genomic interval the sequences are anchored to.
        requested_outputs: Output types to request.
        ontology_terms: Ontology curies, e.g. ["EFO:0002067"].
        chunk_size: Sequences per request.
        max_workers: Initial parallel requests.
        min_workers: Minimum parallel requests after backoff.
        max_retries: Attempts per chunk.
        base_wait: Initial backoff in seconds (exponential, with jitter).
        verbose: Print progress.

    Returns:
        Predictions in input order; None for chunks that never succeeded.
    """
    predictions: list = []
    workers = max_workers
    rng = np.random.default_rng(0)

    for start in range(0, len(sequences), chunk_size):
        chunk = sequences[start:start + chunk_size]
        result = None

        for attempt in range(max_retries):
            try:
                result = model.predict_sequences(
                    sequences=chunk,
                    requested_outputs=requested_outputs,
                    ontology_terms=ontology_terms,
                    max_workers=workers,
                    intervals=[interval] * len(chunk),
                )
                break

            except Exception as error:  # gRPC and API wrapper errors alike
                workers = max(min_workers, workers // 2)
                wait = base_wait * (2 ** attempt) + rng.uniform(0, 5)
                if verbose:
                    print(
                        f"    [!] chunk {start // chunk_size} attempt "
                        f"{attempt + 1}/{max_retries} failed: {error}\n"
                        f"    [*] workers -> {workers}, sleeping {wait:.0f}s",
                        flush=True,
                    )
                if attempt < max_retries - 1:
                    time.sleep(wait)

        if result is None:
            warnings.warn(
                f"Chunk starting at index {start} failed after {max_retries} "
                f"attempts; padding with None."
            )
            predictions.extend([None] * len(chunk))
        else:
            predictions.extend(result)
            # Recover parallelism slowly once the service is responsive again.
            workers = min(max_workers, workers + 1)

    return predictions


# --------------------------------------------------------------------------- #
# 5. Interaction statistics
# --------------------------------------------------------------------------- #

def _as_float(value: Any) -> float:
    """Convert a scalar or single-element sequence to float."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return float(np.mean(value))
    return float(value)


def build_pairwise_interactions(
    scram: pd.DataFrame,
    value_col: str = "ATAC_counts",
    enhancer_col: str = "name",
    scrambled_col: str = "scrambled_seqlets",
    n_scrambled_col: str = "n_scrambled",
    shuffle_col: str = "shuffle_idx",
    seqlets: pd.DataFrame | None = None,
    scale: str = "log",
    pseudocount: float = 1e-6,
) -> pd.DataFrame:
    """Pairwise motif interactions I = delta_AB - delta_A - delta_B, paired by shuffle.

    Args:
        scram: Unaggregated scramble results, one row per sequence.
        value_col: Readout column.
        enhancer_col: Parent enhancer column.
        scrambled_col: Column with the scrambled seqlet labels.
        n_scrambled_col: Column with the number scrambled.
        shuffle_col: Shuffle index column.
        seqlets: Seqlet table; adds spacing_bp and coordinates when given.
        scale: "log" (multiplicative) or "linear" (additive).
        pseudocount: Added before taking the log.

    Returns:
        One row per enhancer and motif pair with interaction, standard error,
        number of shuffles, uncorrected t-test p-value and both main effects.
    """
    if scale not in {"log", "linear"}:
        raise ValueError(f"Unknown scale '{scale}'.")

    def transform(x: float) -> float:
        return np.log(x + pseudocount) if scale == "log" else x

    records = []

    for name, group in scram.groupby(enhancer_col, sort=True):
        group = group.copy()
        group["_value"] = group[value_col].apply(_as_float)
        group["_scr"] = group[scrambled_col].apply(
            lambda v: tuple(sorted(v)) if isinstance(v, (list, np.ndarray)) else ()
        )

        baseline = group[group[n_scrambled_col] == 0]
        if baseline.empty:
            continue
        y0 = transform(baseline["_value"].mean())

        singles = {
            row["_scr"][0]: row
            for _, row in group[group[n_scrambled_col] == 1].iterrows()
        }
        single_series = {
            label: group[
                (group[n_scrambled_col] == 1) & (group["_scr"] == (label,))
            ].set_index(shuffle_col)["_value"]
            for label in singles
        }

        for _, row in group[group[n_scrambled_col] == 2].iterrows():
            a, b = row["_scr"]
            if a not in single_series or b not in single_series:
                continue

            pair_series = group[
                (group[n_scrambled_col] == 2) & (group["_scr"] == (a, b))
            ].set_index(shuffle_col)["_value"]

            joined = pd.concat(
                {"ab": pair_series, "a": single_series[a], "b": single_series[b]},
                axis=1,
            ).dropna()
            if joined.empty:
                continue

            interaction = (
                transform(joined["ab"]) - transform(joined["a"])
                - transform(joined["b"]) + y0
            ).values

            n = len(interaction)
            mean = float(np.mean(interaction))
            sem = float(stats.sem(interaction)) if n > 1 else np.nan
            pval = (
                float(stats.ttest_1samp(interaction, 0.0).pvalue)
                if n > 2 else np.nan
            )

            records.append(dict(
                enhancer=name,
                motif_a=a,
                motif_b=b,
                interaction=mean,
                sem=sem,
                n_shuffles=n,
                pval=pval,
                delta_a=float(np.mean(transform(single_series[a]) - y0)),
                delta_b=float(np.mean(transform(single_series[b]) - y0)),
                delta_ab=float(np.mean(transform(pair_series) - y0)),
                baseline=float(np.exp(y0) if scale == "log" else y0),
                scale=scale,
            ))

    out = pd.DataFrame(records)
    if out.empty:
        return out

    # Deduplicate: each pair appears once per shuffle in the source frame.
    out = out.drop_duplicates(subset=["enhancer", "motif_a", "motif_b"])
    out = out.reset_index(drop=True)

    if seqlets is not None:
        out = _attach_spacing(out, seqlets)
    return out


def _attach_spacing(pairs: pd.DataFrame, seqlets: pd.DataFrame) -> pd.DataFrame:
    """Add the edge-to-edge gap in base pairs between the two motifs of each pair."""
    meta = seqlets.rename(columns={"name": "enhancer", "example_idx": "motif"})
    needed = {"enhancer", "motif", "start_seq", "end_seq"}
    if not needed.issubset(meta.columns):
        warnings.warn("Seqlet table lacks coordinate columns; spacing not attached.")
        return pairs

    coords = meta[list(needed)].drop_duplicates(["enhancer", "motif"])
    out = pairs.copy()

    for side in ("a", "b"):
        out = out.merge(
            coords.rename(columns={
                "motif": f"motif_{side}",
                "start_seq": f"start_{side}",
                "end_seq": f"end_{side}",
            }),
            on=["enhancer", f"motif_{side}"], how="left",
        )

    left_first = out["start_a"] <= out["start_b"]
    out["spacing_bp"] = np.where(
        left_first,
        out["start_b"] - out["end_a"],
        out["start_a"] - out["end_b"],
    )
    return out


def build_motif_effects(
    scram: pd.DataFrame,
    value_col: str = "ATAC_counts",
    enhancer_col: str = "name",
    scrambled_col: str = "scrambled_seqlets",
    n_scrambled_col: str = "n_scrambled",
    shuffle_col: str = "shuffle_idx",
    seqlets: pd.DataFrame | None = None,
    scale: str = "log",
    pseudocount: float = 1e-6,
) -> pd.DataFrame:
    """Main effect of scrambling each motif, paired by shuffle.

    Args:
        scram: Unaggregated scramble results, one row per sequence.
        value_col: Readout column.
        enhancer_col: Parent enhancer column.
        scrambled_col: Column with the scrambled seqlet labels.
        n_scrambled_col: Column with the number scrambled.
        shuffle_col: Shuffle index column.
        seqlets: Seqlet table (name, example_idx, attribution, start_seq, end_seq)
            to attach attribution and position.
        scale: "log" or "linear".
        pseudocount: Added before taking the log.

    Returns:
        One row per enhancer and motif with effect, standard error and
        uncorrected p-value.
    """
    def transform(x: float) -> float:
        return np.log(x + pseudocount) if scale == "log" else x

    records = []

    for name, group in scram.groupby(enhancer_col, sort=True):
        group = group.copy()
        group["_value"] = group[value_col].apply(_as_float)

        baseline = group[group[n_scrambled_col] == 0]
        if baseline.empty:
            continue
        y0 = transform(baseline["_value"].mean())

        singles = group[group[n_scrambled_col] == 1]
        for label, sub in singles.groupby(singles[scrambled_col].apply(lambda v: v[0])):
            values = transform(sub["_value"].to_numpy()) - y0
            n = len(values)
            records.append(dict(
                enhancer=name,
                motif=label,
                delta=float(np.mean(values)),
                sem=float(stats.sem(values)) if n > 1 else np.nan,
                n_shuffles=n,
                pval=float(stats.ttest_1samp(values, 0.0).pvalue) if n > 2 else np.nan,
                baseline=float(np.exp(y0) if scale == "log" else y0),
                scale=scale,
            ))

    out = pd.DataFrame(records)
    if out.empty or seqlets is None:
        return out

    meta = seqlets.rename(columns={"name": "enhancer", "example_idx": "motif"})
    keep = [c for c in ["enhancer", "motif", "attribution", "start_seq", "end_seq"]
            if c in meta.columns]
    return out.merge(meta[keep].drop_duplicates(["enhancer", "motif"]),
                     on=["enhancer", "motif"], how="left")


def build_cooperativity_table(
    scram: pd.DataFrame,
    value_col: str = "ATAC_counts",
    enhancer_col: str = "name",
    scrambled_col: str = "scrambled_seqlets",
    n_scrambled_col: str = "n_scrambled",
    shuffle_col: str = "shuffle_idx",
    scale: str = "log",
    pseudocount: float = 1e-6,
) -> pd.DataFrame:
    """Per-enhancer cooperativity delta_all - sum_i delta_i, paired by shuffle.

    Negative is super-additive, positive sub-additive.

    Args:
        scram: Unaggregated scramble results, one row per sequence.
        value_col: Readout column.
        enhancer_col: Parent enhancer column.
        scrambled_col: Column with the scrambled seqlet labels.
        n_scrambled_col: Column with the number scrambled.
        shuffle_col: Shuffle index column.
        scale: "log" or "linear".
        pseudocount: Added before taking the log.

    Returns:
        One row per enhancer with observed all-scrambled effect, additive
        prediction, cooperativity, standard error and p-value.
    """
    def transform(x: float) -> float:
        return np.log(x + pseudocount) if scale == "log" else x

    rows = []

    for name, group in scram.groupby(enhancer_col, sort=True):
        group = group.copy()
        group["_value"] = group[value_col].apply(_as_float)

        baseline = group[group[n_scrambled_col] == 0]
        singles = group[group[n_scrambled_col] == 1]
        if baseline.empty or singles.empty:
            continue

        n_max = int(group[n_scrambled_col].max())
        if n_max < 2:
            continue
        all_scrambled = group[group[n_scrambled_col] == n_max]

        y0 = transform(baseline["_value"].mean())

        single_series = {
            label: sub.set_index(shuffle_col)["_value"]
            for label, sub in
            singles.groupby(singles[scrambled_col].apply(lambda v: v[0]))
        }
        if len(single_series) != n_max:
            continue  # design incomplete for this enhancer

        joined = pd.concat(
            {"all": all_scrambled.set_index(shuffle_col)["_value"], **single_series},
            axis=1,
        ).dropna()
        if joined.empty:
            continue

        singles_log = transform(joined.drop(columns="all")).sum(axis=1)
        coop_series = transform(joined["all"]) - singles_log + (n_max - 1) * y0

        n = len(coop_series)
        rows.append(dict(
            enhancer=name,
            n_seqlets=n_max,
            additive_prediction=float(
                (transform(joined.drop(columns="all")) - y0).sum(axis=1).mean()
            ),
            observed=float((transform(joined["all"]) - y0).mean()),
            cooperativity=float(coop_series.mean()),
            sem=float(stats.sem(coop_series)) if n > 1 else np.nan,
            n_shuffles=n,
            pval=float(stats.ttest_1samp(coop_series, 0.0).pvalue) if n > 2 else np.nan,
            scale=scale,
        ))

    return pd.DataFrame(rows)


def build_saturation_split(
    scram: pd.DataFrame,
    value_col: str = "ATAC_counts",
    enhancer_col: str = "name",
    scrambled_col: str = "scrambled_seqlets",
    n_scrambled_col: str = "n_scrambled",
    shuffle_col: str = "shuffle_idx",
    scale: str = "log",
    pseudocount: float = 1e-6,
    random_state: int = 0,
) -> pd.DataFrame:
    """Saturation check with main effects and interaction estimated from disjoint shuffle halves.

    Splitting removes the coupling of plotting I against delta_A + delta_B when
    both are estimated from the same shuffles.

    Args:
        scram: Unaggregated scramble results, one row per sequence.
        value_col: Readout column.
        enhancer_col: Parent enhancer column.
        scrambled_col: Column with the scrambled seqlet labels.
        n_scrambled_col: Column with the number scrambled.
        shuffle_col: Shuffle index column.
        scale: "log" or "linear".
        pseudocount: Added before taking the log.
        random_state: Seed for assigning shuffles to halves.

    Returns:
        One row per enhancer and pair with additive_split (first half),
        interaction_split (second half), full-data equivalents and log_baseline.
    """
    def transform(x: float) -> float:
        return np.log(x + pseudocount) if scale == "log" else x

    rng = np.random.default_rng(random_state)
    records = []

    for name, group in scram.groupby(enhancer_col, sort=True):
        group = group.copy()
        group["_value"] = group[value_col].apply(_as_float)
        group["_scr"] = group[scrambled_col].apply(
            lambda v: tuple(sorted(v)) if isinstance(v, (list, np.ndarray)) else ()
        )

        baseline = group[group[n_scrambled_col] == 0]
        if baseline.empty:
            continue
        y0 = transform(baseline["_value"].mean())

        shuffles = sorted(group[shuffle_col].dropna().unique())
        if len(shuffles) < 4:
            continue
        shuffled = rng.permutation(shuffles)
        half = len(shuffled) // 2
        fold_x = set(shuffled[:half])
        fold_i = set(shuffled[half:])

        def series_for(key: tuple, folds: set | None = None) -> pd.Series:
            sub = group[group["_scr"] == key]
            if folds is not None:
                sub = sub[sub[shuffle_col].isin(folds)]
            return sub.set_index(shuffle_col)["_value"]

        singles = {key[0] for key in group["_scr"] if len(key) == 1}

        for key in {k for k in group["_scr"] if len(k) == 2}:
            a, b = key
            if a not in singles or b not in singles:
                continue

            # x axis: main effects from the first half only.
            a_x, b_x = series_for((a,), fold_x), series_for((b,), fold_x)
            if a_x.empty or b_x.empty:
                continue
            additive_split = float(
                (transform(a_x) - y0).mean() + (transform(b_x) - y0).mean()
            )

            # y axis: full interaction contrast from the second half only.
            joined = pd.concat({"ab": series_for(key, fold_i),
                                "a": series_for((a,), fold_i),
                                "b": series_for((b,), fold_i)}, axis=1).dropna()
            if joined.empty:
                continue
            interaction_split = float(
                (transform(joined["ab"]) - transform(joined["a"])
                 - transform(joined["b"]) + y0).mean()
            )

            # Full-data versions, for the side-by-side comparison.
            full = pd.concat({"ab": series_for(key), "a": series_for((a,)),
                              "b": series_for((b,))}, axis=1).dropna()
            additive_full = float(
                (transform(series_for((a,))) - y0).mean()
                + (transform(series_for((b,))) - y0).mean()
            )
            interaction_full = float(
                (transform(full["ab"]) - transform(full["a"])
                 - transform(full["b"]) + y0).mean()
            ) if not full.empty else np.nan

            records.append(dict(
                enhancer=name, motif_a=a, motif_b=b,
                additive_split=additive_split,
                interaction_split=interaction_split,
                additive_full=additive_full,
                interaction_full=interaction_full,
                log_baseline=float(y0),
                n_fold_x=len(fold_x), n_fold_i=len(fold_i),
                scale=scale,
            ))

    return pd.DataFrame(records)


def build_floor_check(
    scram: pd.DataFrame,
    value_col: str = "ATAC_counts",
    enhancer_col: str = "name",
    scrambled_col: str = "scrambled_seqlets",
    n_scrambled_col: str = "n_scrambled",
    floor: float | None = None,
    floor_percentile: float = 0.1,
    pseudocount: float = 1e-6,
) -> pd.DataFrame:
    """Compare each pair's multiplicative null prediction with the readout floor.

    Args:
        scram: Unaggregated scramble results, one row per sequence.
        value_col: Readout column.
        enhancer_col: Parent enhancer column.
        scrambled_col: Column with the scrambled seqlet labels.
        n_scrambled_col: Column with the number scrambled.
        floor: Floor value; estimated from the data when None.
        floor_percentile: Percentile of all predictions used as the floor.
        pseudocount: Added before taking the log.

    Returns:
        One row per enhancer and pair with null prediction, observed value, floor
        and null_below_floor.
    """
    all_values = scram[value_col].apply(_as_float).dropna()
    if floor is None:
        floor = float(np.percentile(all_values, floor_percentile))

    records = []

    for name, group in scram.groupby(enhancer_col, sort=True):
        group = group.copy()
        group["_value"] = group[value_col].apply(_as_float)
        group["_scr"] = group[scrambled_col].apply(
            lambda v: tuple(sorted(v)) if isinstance(v, (list, np.ndarray)) else ()
        )

        baseline = group[group[n_scrambled_col] == 0]
        if baseline.empty:
            continue
        y0 = float(baseline["_value"].mean())

        means = group.groupby("_scr")["_value"].mean().to_dict()
        singles = {key[0]: value for key, value in means.items() if len(key) == 1}

        for key, observed in means.items():
            if len(key) != 2:
                continue
            a, b = key
            if a not in singles or b not in singles:
                continue

            # Multiplicative null: the product of the two fold changes.
            null_prediction = y0 * (singles[a] / y0) * (singles[b] / y0)
            records.append(dict(
                enhancer=name, motif_a=a, motif_b=b,
                baseline=y0,
                null_prediction=float(null_prediction),
                observed=float(observed),
                floor=floor,
                null_below_floor=bool(null_prediction < floor),
                headroom=float(np.log((null_prediction + pseudocount)
                                      / (floor + pseudocount))),
            ))

    return pd.DataFrame(records)


def build_dose_response(
    scram: pd.DataFrame,
    value_col: str = "ATAC_counts",
    enhancer_col: str = "name",
    scrambled_col: str = "scrambled_seqlets",
    n_scrambled_col: str = "n_scrambled",
    seqlets: pd.DataFrame | None = None,
    scale: str = "log",
    pseudocount: float = 1e-6,
) -> pd.DataFrame:
    """Observed effect against the multiplicative null for every scrambled combination.

    Args:
        scram: Unaggregated scramble results, one row per sequence.
        value_col: Readout column.
        enhancer_col: Parent enhancer column.
        scrambled_col: Column with the scrambled seqlet labels.
        n_scrambled_col: Column with the number scrambled.
        seqlets: Seqlet table (name, example_idx, start_seq, end_seq); adds the
            scrambled width in base pairs.
        scale: "log" or "linear".
        pseudocount: Added before taking the log.

    Returns:
        One row per enhancer and combination with additive prediction, observed
        effect, difference, order and optionally scrambled width.
    """
    def transform(x: float) -> float:
        return np.log(x + pseudocount) if scale == "log" else x

    widths = {}
    if seqlets is not None and {"start_seq", "end_seq"}.issubset(seqlets.columns):
        for _, row in seqlets.iterrows():
            widths[(row["name"], row["example_idx"])] = (
                int(row["end_seq"]) - int(row["start_seq"])
            )

    records = []

    for name, group in scram.groupby(enhancer_col, sort=True):
        group = group.copy()
        group["_value"] = group[value_col].apply(_as_float)
        group["_scr"] = group[scrambled_col].apply(
            lambda v: tuple(sorted(v)) if isinstance(v, (list, np.ndarray)) else ()
        )

        baseline = group[group[n_scrambled_col] == 0]
        if baseline.empty:
            continue
        y0 = transform(baseline["_value"].mean())

        means = group.groupby("_scr")["_value"].mean().apply(transform).to_dict()
        singles = {key[0]: value - y0
                   for key, value in means.items() if len(key) == 1}

        for key, value in means.items():
            if not key or any(m not in singles for m in key):
                continue

            predicted = float(sum(singles[m] for m in key))
            observed = float(value - y0)
            scrambled_bp = (sum(widths.get((name, m), np.nan) for m in key)
                            if widths else np.nan)

            records.append(dict(
                enhancer=name,
                combination=",".join(key),
                n_scrambled=len(key),
                predicted=predicted,
                observed=observed,
                deviation=observed - predicted,
                scrambled_bp=scrambled_bp,
                baseline=float(np.exp(y0) if scale == "log" else y0),
                scale=scale,
            ))

    return pd.DataFrame(records)


def partial_correlation(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray,
                        z: pd.Series | np.ndarray) -> tuple[float, float, int]:
    """Pearson correlation of x and y after removing the linear effect of z.

    Args:
        x: First variable.
        y: Second variable.
        z: Control variable.

    Returns:
        Tuple of (partial correlation, two-sided p-value, sample size).
    """
    frame = pd.DataFrame(dict(x=x, y=y, z=z)).dropna()
    n = len(frame)
    if n < 4 or frame["z"].nunique() < 2:
        return np.nan, np.nan, n

    r_xy = stats.pearsonr(frame["x"], frame["y"])[0]
    r_xz = stats.pearsonr(frame["x"], frame["z"])[0]
    r_yz = stats.pearsonr(frame["y"], frame["z"])[0]

    denominator = np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))
    if denominator <= 0:
        return np.nan, np.nan, n

    partial = (r_xy - r_xz * r_yz) / denominator
    partial = float(np.clip(partial, -1.0, 1.0))

    # One covariate removed, so two degrees of freedom are lost.
    degrees = n - 3
    if degrees <= 0 or abs(partial) >= 1:
        return partial, np.nan, n
    t_stat = partial * np.sqrt(degrees / (1 - partial ** 2))
    pval = float(2 * stats.t.sf(abs(t_stat), degrees))
    return partial, pval, n


def add_bh_correction(
    df: pd.DataFrame,
    pval_col: str = "pval",
    out_col: str = "qval",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Add Benjamini-Hochberg adjusted p-values across all rows of a table.

    Args:
        df: Table with a p-value column.
        pval_col: p-value column.
        out_col: Name of the adjusted p-value column.
        alpha: Threshold for the significant flag.

    Returns:
        Copy of df with adjusted p-values and a boolean significant column;
        missing p-values stay missing.
    """
    out = df.copy()
    mask = out[pval_col].notna()
    out[out_col] = np.nan

    if mask.sum():
        pvals = out.loc[mask, pval_col].to_numpy()
        order = np.argsort(pvals)
        ranked = pvals[order]
        n = len(ranked)

        adjusted = ranked * n / np.arange(1, n + 1)
        # Enforce monotonicity from the largest p-value downwards.
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        adjusted = np.clip(adjusted, 0, 1)

        restored = np.empty(n)
        restored[order] = adjusted
        out.loc[mask, out_col] = restored

    out["significant"] = out[out_col] < alpha
    return out