"""Generate an HTML report with various QC charts for of MS²Rescore results."""

import importlib.resources
import json
import logging
from datetime import datetime
from itertools import cycle
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import plotly.express as px
import psm_utils.io
from jinja2 import Environment, FileSystemLoader
from psm_utils.psm_list import PSMList

try:
    import tomllib
except ImportError:
    import tomli as tomllib

import ms2rescore
import ms2rescore.report.charts as charts
import ms2rescore.report.templates as templates
from ms2rescore.report.utils import (
    get_confidence_estimates,
    get_feature_values,
    read_feature_names,
)

logger = logging.getLogger(__name__)

PLOTLY_HTML_KWARGS = {
    "full_html": False,
    "include_plotlyjs": False,
    "include_mathjax": False,
    "config": {
        "displayModeBar": True,
        "displaylogo": False,
    },
}


TEXTS = tomllib.loads(importlib.resources.read_text(templates, "texts.toml"))


def generate_report(
    output_path_prefix: str,
    psm_list: Optional[psm_utils.PSMList] = None,
    feature_names: Optional[Dict[str, list]] = None,
    use_txt_log: bool = False,
):
    """
    Generate the report.

    Parameters
    ----------
    output_path_prefix
        Prefix of the MS²Rescore output file names. For example, if the PSM file is
        ``/path/to/file.psms.tsv``, the prefix is ``/path/to/file.ms2rescore``.
    psm_list
        PSMs to be used for the report. If not provided, the PSMs will be read from the
        PSM file that matches the ``output_path_prefix``.
    feature_names
        Feature names to be used for the report. If not provided, the feature names will be
        read from the feature names file that matches the ``output_path_prefix``.
    use_txt_log
        If True, the log file will be read from ``output_path_prefix + ".log.txt"`` instead of
        ``output_path_prefix + ".log.html"``.

    """
    files = _collect_files(output_path_prefix, use_txt_log=use_txt_log)

    # Read PSMs
    if not psm_list:
        if files["PSMs"]:
            logger.info("Reading PSMs...")
            psm_list = psm_utils.io.read_file(files["PSMs"], filetype="tsv", show_progressbar=True)
        else:
            raise FileNotFoundError("PSM file not found and no PSM list provided.")

    # Read config
    config = json.loads(files["configuration"].read_text())

    logger.debug("Recalculating confidence estimates...")
    fasta_file = config["ms2rescore"]["fasta_file"]
    confidence_before, confidence_after = get_confidence_estimates(psm_list, fasta_file)

    overview_context = _get_overview_context(confidence_before, confidence_after)
    target_decoy_context = _get_target_decoy_context(psm_list)
    features_context = _get_features_context(psm_list, files, feature_names=feature_names)
    config_context = _get_config_context(config)
    log_context = _get_log_context(files)

    context = {
        "metadata": {
            "generated_on": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "ms2rescore_version": ms2rescore.__version__,  # TODO: Write during run?
            "psm_filename": "\n".join(
                [Path(id_file).name for id_file in config["ms2rescore"]["psm_file"]]
            ),
        },
        "main_tabs": [
            {
                "id": "main_tab_comparison",
                "title": "Overview",
                "template": "overview.html",
                "context": overview_context,
            },
            {
                "id": "main_tab_target_decoy",
                "title": "Target/decoy evaluation",
                "template": "target-decoy.html",
                "context": target_decoy_context,
            },
            {
                "id": "main_tab_features",
                "title": "Rescoring features",
                "template": "features.html",
                "context": features_context,
            },
            {
                "id": "main_tab_config",
                "title": "Full configuration",
                "template": "config.html",
                "context": config_context,
            },
            {
                "id": "main_tab_log",
                "title": "Log",
                "template": "log.html",
                "context": log_context,
            },
        ],
    }

    _render_and_write(output_path_prefix, **context)


def _collect_files(output_path_prefix, use_txt_log=False):
    """Collect all files generated by MS²Rescore."""
    logger.debug("Collecting files...")
    files = {
        "PSMs": Path(output_path_prefix + ".psms.tsv").resolve(),
        "configuration": Path(output_path_prefix + ".full-config.json").resolve(),
        "feature names": Path(output_path_prefix + ".feature_names.tsv").resolve(),
        "feature weights": Path(output_path_prefix + ".mokapot.weights.tsv").resolve(),
        "log": Path(output_path_prefix + ".log.txt").resolve()
        if use_txt_log
        else Path(output_path_prefix + ".log.html").resolve(),
    }
    for file, path in files.items():
        if Path(path).is_file():
            logger.debug("✅ Found %s: '%s'", file, path.as_posix())
        else:
            logger.warning("❌ %s: '%s'", file, path.as_posix())
            files[file] = None
    return files


def _get_stats_context(confidence_before, confidence_after):
    """Return context for overview statistics pane."""
    stats = []
    levels = ["psms", "peptides", "proteins"]
    level_names = ["PSMs", "Peptides", "Protein groups"]
    card_colors = ["card-bg-blue", "card-bg-green", "card-bg-red"]

    # Cannot report stats if confidence estimates are not present
    if not confidence_before or not confidence_after:
        return stats

    for level, level_name, card_color in zip(levels, level_names, card_colors):
        try:
            before = confidence_before.accepted[level.lower()]
            after = confidence_after.accepted[level.lower()]
        except KeyError:
            continue  # Level not present (e.g. no fasta provided)
        if not before or not after:
            continue
        increase = (after - before) / before * 100
        stats.append(
            {
                "item": level_name,
                "card_color": card_color,
                "number": after,
                "diff": f"({after - before:+})",
                "percentage": f"{increase:.1f}%",
                "is_increase": increase > 0,
                "bar_percentage": before / after * 100 if increase > 0 else after / before * 100,
                "bar_color": "#24a143" if increase > 0 else "#a12424",
            }
        )
    return stats


def _get_overview_context(confidence_before, confidence_after) -> dict:
    """Return context for overview tab."""
    logger.debug("Generating overview charts...")
    return {
        "stats": _get_stats_context(confidence_before, confidence_after),
        "charts": [
            {
                "title": TEXTS["charts"]["score_comparison"]["title"],
                "description": TEXTS["charts"]["score_comparison"]["description"],
                "chart": charts.score_scatter_plot(
                    confidence_before,
                    confidence_after,
                ).to_html(**PLOTLY_HTML_KWARGS),
            },
            {
                "title": TEXTS["charts"]["fdr_comparison"]["title"],
                "description": TEXTS["charts"]["fdr_comparison"]["description"],
                "chart": charts.fdr_plot_comparison(
                    confidence_before,
                    confidence_after,
                ).to_html(**PLOTLY_HTML_KWARGS),
            },
            {
                "title": TEXTS["charts"]["identification_overlap"]["title"],
                "description": TEXTS["charts"]["identification_overlap"]["description"],
                "chart": charts.identification_overlap(
                    confidence_before,
                    confidence_after,
                ).to_html(**PLOTLY_HTML_KWARGS),
            },
        ],
    }


def _get_target_decoy_context(psm_list) -> dict:
    logger.debug("Generating target-decoy charts...")
    psm_df = psm_list.to_dataframe()
    return {
        "charts": [
            {
                "title": TEXTS["charts"]["score_histogram"]["title"],
                "description": TEXTS["charts"]["score_histogram"]["description"],
                "chart": charts.score_histogram(psm_df).to_html(**PLOTLY_HTML_KWARGS),
            },
            {
                "title": TEXTS["charts"]["pp_plot"]["title"],
                "description": TEXTS["charts"]["pp_plot"]["description"],
                "chart": charts.pp_plot(psm_df).to_html(**PLOTLY_HTML_KWARGS),
            },
        ]
    }


def _get_features_context(
    psm_list: PSMList,
    files: Dict[str, Path],
    feature_names: Optional[Dict[str, list]] = None,
) -> dict:
    """Return context for features tab."""
    logger.debug("Generating feature-related charts...")
    context = {"charts": []}

    # Get feature names, mapping with generator, and flat list
    if not feature_names:
        feature_names = read_feature_names(files["feature names"])
    feature_names_flat = [f_name for f_list in feature_names.values() for f_name in f_list]
    feature_names_inv = {name: gen for gen, f_list in feature_names.items() for name in f_list}

    # Get fixed color map for feature generators
    color_map = dict(zip(feature_names.keys(), cycle(px.colors.qualitative.Plotly)))

    # feature weights
    if not files["feature weights"]:
        logger.warning("Could not find feature weights files. Skipping feature weights plot.")
    else:
        feature_weights = pd.read_csv(files["feature weights"], sep="\t").melt(
            var_name="feature", value_name="weight"
        )
        feature_weights["feature"] = feature_weights["feature"].str.replace(
            r"^(feature:)?", "", regex=True
        )
        feature_weights["feature_generator"] = feature_weights["feature"].map(feature_names_inv)

        context["charts"].append(
            {
                "title": TEXTS["charts"]["feature_usage"]["title"],
                "description": TEXTS["charts"]["feature_usage"]["description"],
                "chart": charts.feature_weights_by_generator(
                    feature_weights, color_discrete_map=color_map
                ).to_html(**PLOTLY_HTML_KWARGS)
                + charts.feature_weights(feature_weights, color_discrete_map=color_map).to_html(
                    **PLOTLY_HTML_KWARGS
                ),
            }
        )

    # Individual feature performance
    features = get_feature_values(psm_list, feature_names_flat)
    _, feature_ecdf_auc = charts.calculate_feature_qvalues(features, psm_list["is_decoy"])
    feature_ecdf_auc["feature_generator"] = feature_ecdf_auc["feature"].map(feature_names_inv)

    context["charts"].append(
        {
            "title": TEXTS["charts"]["feature_performance"]["title"],
            "description": TEXTS["charts"]["feature_performance"]["description"],
            "chart": charts.feature_ecdf_auc_bar(
                feature_ecdf_auc, color_discrete_map=color_map
            ).to_html(**PLOTLY_HTML_KWARGS),
        }
    )

    # MS²PIP specific charts
    if "ms2pip" in feature_names and "spec_pearson_norm" in feature_names["ms2pip"]:
        context["charts"].append(
            {
                "title": TEXTS["charts"]["ms2pip_pearson"]["title"],
                "description": TEXTS["charts"]["ms2pip_pearson"]["description"],
                "chart": charts.ms2pip_correlation(
                    features, psm_list["is_decoy"], psm_list["qvalue"]
                ).to_html(**PLOTLY_HTML_KWARGS),
            }
        )

    # DeepLC specific charts
    if "deeplc" in feature_names:
        import deeplc.plot

        scatter_chart = deeplc.plot.scatter(
            df=features[
                (psm_list["is_decoy"] == False) & (psm_list["qvalue"] <= 0.01)
            ],  # noqa: E712
            predicted_column="predicted_retention_time_best",
            observed_column="observed_retention_time_best",
        )
        baseline_chart = deeplc.plot.distribution_baseline(
            df=features[
                (psm_list["is_decoy"] == False) & (psm_list["qvalue"] <= 0.01)
            ],  # noqa: E712
            predicted_column="predicted_retention_time_best",
            observed_column="observed_retention_time_best",
        )
        context["charts"].append(
            {
                "title": TEXTS["charts"]["deeplc_performance"]["title"],
                "description": TEXTS["charts"]["deeplc_performance"]["description"],
                "chart": scatter_chart.to_html(**PLOTLY_HTML_KWARGS)
                + baseline_chart.to_html(**PLOTLY_HTML_KWARGS),
            }
        )

    return context


def _get_config_context(config: dict) -> dict:
    """Return context for config tab."""
    return {
        "description": TEXTS["configuration"]["description"],
        "config": json.dumps(config, indent=4),
    }


def _get_log_context(files: Dict[str, Path]) -> dict:
    """Return context for log tab."""
    if not files["log"]:
        return {"log": "<i>Log file could not be found.</i>"}

    if files["log"].suffix == ".html":
        return {"log": files["log"].read_text(encoding="utf-8")}

    if files["log"].suffix == ".txt":
        return {"log": "<pre><code>" + files["log"].read_text(encoding="utf-8") + "</code></pre>"}


def _render_and_write(output_path_prefix: str, **context):
    """Render template with context and write to HTML file."""
    report_path = Path(output_path_prefix + ".report.html").resolve()
    logger.info("Writing report to %s", report_path.as_posix())
    template_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(template_dir, encoding="utf-8"))
    template = env.get_template("base.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(template.render(**context))
