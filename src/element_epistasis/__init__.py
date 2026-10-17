# Code by Jakob Then 2026
# Steinmetz Lab, Genome Biology Unit, EMBL Heidelberg

"""In silico enhancer and TF motif epistasis with AlphaGenome.

Modules:
    glm, GLM_fit           log-link interaction GLM (generic / enhancer-level)
    enhancer_analysis      enhancer-level tables, validation, interdependency
    tf_cooperativity       seqlet design, prediction driver, motif tables
    cross_level            motif-level to enhancer-level and screen links
    report, cross_report   text summaries; pipeline runs report end to end
    *_plots, plot_style    figures and shared style
    ism, paths             ISM file I/O and project paths

Sign convention: effects are log fold changes against the unperturbed sequence;
an interaction below zero is super-additive, above zero sub-additive.
"""

__all__ = [
    "plot_style",
    "tf_cooperativity",
    "tf_plots",
    "enhancer_analysis",
    "enhancer_plots",
    "report",
    "pipeline",
    "glm",
    "GLM_fit",
    "cross_plots",
    "cross_report",
    "cross_level",
    "ism",
    "paths",
]
