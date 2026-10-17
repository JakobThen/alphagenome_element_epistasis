"""Single-call driver that builds every analysis table and writes the text summary.

All inputs are optional, so the TF level, the enhancer level or both can be run.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import pandas as pd

from element_epistasis import enhancer_analysis as ea
from element_epistasis import tf_cooperativity as tc
from element_epistasis.report import write_summary


def _try(label: str, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run a builder and return None with a warning if it raises."""
    try:
        return function(*args, **kwargs)
    except Exception as error:
        warnings.warn(f"{label} failed: {type(error).__name__}: {error}")
        return None


def run_all(
    *,
    # TF level
    tf_scram: pd.DataFrame | None = None,
    tf_seqlets: pd.DataFrame | None = None,
    tf_control_scram: pd.DataFrame | None = None,
    tf_control_windows: pd.DataFrame | None = None,
    tf_atac_col: str = "ATAC_counts",
    tf_rna_col: str = "RNA_counts",
    # Enhancer level
    enh_coef: pd.DataFrame | None = None,
    enh_atac_coef: pd.DataFrame | None = None,
    enh_scram: pd.DataFrame | None = None,
    enh_n_scrambled_col: str = "n_scrambled",
    enh_rna_col: str = "RNA_counts",
    enh_atac_col: str = "ATAC_counts",
    enhancers: pd.DataFrame | None = None,
    single_comp: pd.DataFrame | None = None,
    pair_comp: pd.DataFrame | None = None,
    insilico_p_col: str | None = None,
    model_results: dict | None = None,
    # Output
    summary_path: str | None = None,
    alpha: float = 0.05,
    top_n: int = 10,
    verbose: bool = True,
) -> dict:
    """Build every analysis table from the raw frames and optionally write the summary.

    Args:
        tf_scram: Unaggregated TF scramble results, one row per sequence.
        tf_seqlets: Seqlet table with coordinates and attribution.
        tf_control_scram: Scramble results of the control windows.
        tf_control_windows: Control window table.
        tf_atac_col: Accessibility column in the TF frames.
        tf_rna_col: Expression column in the TF frames.
        enh_coef: Enhancer-level GLM coefficients on RNA, intercept included.
        enh_atac_coef: Enhancer-level interactions on ATAC, term or enhA/enhB layout.
        enh_scram: Enhancer-level scramble results, for the model-free tables.
        enh_n_scrambled_col: Column with the number of scrambled units.
        enh_rna_col: Expression column in the enhancer frame.
        enh_atac_col: Accessibility column in the enhancer frame.
        enhancers: Enhancer coordinates, for distance annotation.
        single_comp: Single-enhancer comparison with CRISPRi.
        pair_comp: Enhancer-pair comparison with CRISPRi.
        insilico_p_col: In silico significance column; auto-detected when None.
        model_results: Mapping of setting label to fit_all_genes output.
        summary_path: Output path of the text summary; not written when None.
        alpha: Significance threshold.
        top_n: Number of named extremes per direction in the summary.
        verbose: Print progress.

    Returns:
        Dictionary of every table that could be built, keyed as
        report.build_summary expects.
    """
    tables: dict = {}

    def note(message: str) -> None:
        if verbose:
            print(message)

    # ---------------------------------------------------------------- TF level
    if tf_scram is not None:
        note("TF level:")
        tables["motifs"] = _try(
            "build_motif_effects", tc.build_motif_effects,
            tf_scram, value_col=tf_atac_col, seqlets=tf_seqlets)
        if tables.get("motifs") is not None:
            tables["motifs"] = tc.add_bh_correction(tables["motifs"])
            note(f"  motifs            {len(tables['motifs'])}")

        tables["pairs"] = _try(
            "build_pairwise_interactions", tc.build_pairwise_interactions,
            tf_scram, value_col=tf_atac_col, seqlets=tf_seqlets)
        if tables.get("pairs") is not None:
            tables["pairs"] = tc.add_bh_correction(tables["pairs"])
            note(f"  pairs             {len(tables['pairs'])}")

        for key, column in [("coop_atac", tf_atac_col), ("coop_rna", tf_rna_col)]:
            if column not in tf_scram.columns:
                continue
            built = _try(f"build_cooperativity_table[{column}]",
                         tc.build_cooperativity_table,
                         tf_scram, value_col=column)
            if built is not None:
                tables[key] = tc.add_bh_correction(built)
                note(f"  {key:<17} {len(tables[key])}")

        tables["split"] = _try("build_saturation_split",
                               tc.build_saturation_split,
                               tf_scram, value_col=tf_atac_col)
        tables["floor_df"] = _try("build_floor_check", tc.build_floor_check,
                                  tf_scram, value_col=tf_atac_col)
        tables["dose"] = _try("build_dose_response", tc.build_dose_response,
                              tf_scram, value_col=tf_atac_col,
                              seqlets=tf_seqlets)

    if tf_control_scram is not None:
        tables["controls"] = _try(
            "build_motif_effects[controls]", tc.build_motif_effects,
            tf_control_scram, value_col=tf_atac_col, seqlets=tf_control_windows)
        if tables.get("controls") is not None:
            note(f"  controls          {len(tables['controls'])}")

    # ---------------------------------------------------------- Enhancer level
    if enh_coef is not None:
        note("Enhancer level:")
        split_result = _try("split_coef_table", ea.split_coef_table, enh_coef)
        if split_result is not None:
            mains, inters = split_result
            tables["enh_mains"] = tc.add_bh_correction(mains)
            inters = tc.add_bh_correction(inters)
            # attach_main_effects needs the unsplit table for the intercept.
            attached = _try("attach_main_effects", ea.attach_main_effects,
                            inters, enh_coef)
            if attached is not None:
                inters = attached
            if enhancers is not None:
                annotated = _try("annotate_pair_distance",
                                 ea.annotate_pair_distance, inters, enhancers)
                if annotated is not None:
                    inters = annotated
            tables["enh_inters"] = inters
            note(f"  main effects      {len(tables['enh_mains'])}")
            note(f"  interactions      {len(tables['enh_inters'])}")

    if enh_atac_coef is not None:
        built = _try("normalise_interaction_table",
                     ea.normalise_interaction_table, enh_atac_coef)
        if built is not None:
            tables["atac_inters"] = tc.add_bh_correction(built)
            note(f"  ATAC interactions {len(tables['atac_inters'])}")

    if enh_scram is not None:
        tables["enh_coop"] = _try(
            "build_coop_table", ea.build_coop_table, enh_scram,
            value_col=enh_rna_col, n_scrambled_col=enh_n_scrambled_col)
        tables["enh_epi"] = _try(
            "build_pairwise_epistasis", ea.build_pairwise_epistasis, enh_scram,
            value_col=enh_rna_col, n_scrambled_col=enh_n_scrambled_col)

        # Accessibility is vector valued, so it must be expanded per locus.
        if enh_atac_col in enh_scram.columns:
            long = _try("expand_locus_values", ea.expand_locus_values,
                        enh_scram, enh_atac_col)
            if long is not None:
                tables["enh_long"] = long
                built = _try("build_interdependency", ea.build_interdependency,
                             long, value_col=enh_atac_col)
                if built is not None:
                    tables["interdependency"] = tc.add_bh_correction(built)
                    note(f"  interdependency   {len(tables['interdependency'])}")
                tables["coop_atac_locus"] = _try(
                    "build_coop_table[locus]", ea.build_coop_table, long,
                    name_col="locus", value_col=enh_atac_col,
                    n_scrambled_col=enh_n_scrambled_col)
                tables["coop_atac_gene"] = _try(
                    "build_coop_table[gene, summed]", ea.build_coop_table,
                    ea.aggregate_locus_values(enh_scram, enh_atac_col),
                    value_col=enh_atac_col,
                    n_scrambled_col=enh_n_scrambled_col)

                # Trans crosstalk, aggregated at both resolutions.
                if tables.get("interdependency") is not None:
                    tables["trans_by_gene"] = _try(
                        "summarise_trans_effects[gene]",
                        ea.summarise_trans_effects,
                        tables["interdependency"], by="gene")
                    tables["trans_by_locus"] = _try(
                        "summarise_trans_effects[locus]",
                        ea.summarise_trans_effects,
                        tables["interdependency"], by="perturbed")
                    tables["pair_trans"] = _try(
                        "summarise_pair_trans", ea.summarise_pair_trans,
                        tables["interdependency"])

    if single_comp is not None:
        tables["single_comp"] = single_comp
    if pair_comp is not None:
        if tables.get("pair_trans") is not None:
            joined = _try("join_pair_trans", ea.join_pair_trans,
                          pair_comp, tables["pair_trans"])
            if joined is not None:
                pair_comp = joined
                attached = int(joined["trans_strongest"].notna().sum())
                note(f"  pair trans attached to {attached}/{len(joined)} "
                     f"comparison rows")
        tables["pair_comp"] = pair_comp

    # Validation summaries: all effects, and those the in silico analysis calls
    # significant. The screen is not used to define subsets.
    summaries = {}
    for label, comp in [("single enhancers", single_comp),
                        ("enhancer pairs", pair_comp)]:
        if comp is None or comp.empty:
            continue
        insilico_p = next((c for c in [insilico_p_col, "qval", "pval"]
                           if c and c in comp.columns), None)
        built = _try(f"summarise_validation[{label}]", ea.summarise_validation,
                     comp, insilico_p_col=insilico_p, alpha=alpha)
        if built is not None:
            summaries[label] = built
            note(f"  validation {label}: n={built['n']}, "
                 f"in-silico-significant={built.get('n_insilico', 0)}")
    if summaries:
        tables["validation_summaries"] = summaries
    if model_results is not None:
        tables["comparison"] = _try("build_model_comparison",
                                    ea.build_model_comparison, model_results)

    # ----------------------------------------------------------------- Summary
    if summary_path:
        accepted = {
            "motifs", "controls", "pairs", "coop_atac", "coop_rna", "split",
            "floor_df", "dose", "enh_mains", "enh_inters", "atac_inters",
            "enh_coop", "single_comp", "pair_comp", "interdependency",
            "comparison", "cross", "validation_summaries", "trans_by_gene",
            "trans_by_locus", "coop_atac_gene", "coop_atac_locus",
        }
        write_summary(summary_path, alpha=alpha, top_n=top_n,
                      **{k: v for k, v in tables.items() if k in accepted})

    return tables