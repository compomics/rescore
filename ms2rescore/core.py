import json
import logging
from multiprocessing import cpu_count
from typing import Dict, Optional

import psm_utils.io
from mokapot.dataset import LinearPsmDataset
from psm_utils import PSMList

from ms2rescore import exceptions
from ms2rescore.feature_generators import FEATURE_GENERATORS
from ms2rescore.parse_psms import parse_psms
from ms2rescore.parse_spectra import add_precursor_values
from ms2rescore.report import generate
from ms2rescore.rescoring_engines import mokapot, percolator
from ms2rescore.rescoring_engines.mokapot import (
    add_peptide_confidence,
    add_psm_confidence,
)
from ms2rescore.utils import filter_mumble_psms

logger = logging.getLogger(__name__)


def rescore(configuration: Dict, psm_list: Optional[PSMList] = None) -> None:
    """
    Run full MS²Rescore workflow with passed configuration.

    Parameters
    ----------
    configuration
        Dictionary containing ms2rescore configuration.
    psm_list
        PSMList object containing PSMs. If None, PSMs will be read from configuration ``psm_file``.

    """
    logger.debug(
        f"Running MS²Rescore with following configuration: {json.dumps(configuration, indent=4)}"
    )
    config = configuration["ms2rescore"]
    output_file_root = config["output_path"]

    # Write full configuration including defaults to file
    with open(output_file_root + ".full-config.json", "w") as f:
        json.dump(configuration, f, indent=4)

    logger.debug("Using %i of %i available CPUs.", int(config["processes"]), int(cpu_count()))

    # Parse PSMs
    psm_list = parse_psms(config, psm_list)

    # Log #PSMs identified before rescoring
    id_psms_before = _log_id_psms_before(psm_list, max_rank=config["max_psm_rank_output"])

    # Define feature names; get existing feature names from PSM file
    feature_names = dict()
    psm_list_feature_names = {
        feature_name
        for psm_list_features in psm_list["rescoring_features"]
        for feature_name in psm_list_features.keys()
    }
    feature_names["psm_file"] = psm_list_feature_names
    logger.debug(
        f"PSMs already contain the following rescoring features: {psm_list_feature_names}"
    )

    # Add missing precursor info from spectrum file if needed
    available_ms_data = add_precursor_values(
        psm_list, config["spectrum_path"], config["spectrum_id_pattern"]
    )

    # Add rescoring features
    for fgen_name, fgen_config in config["feature_generators"].items():
        # Compile configuration
        conf = config.copy()
        conf.update(fgen_config)
        fgen = FEATURE_GENERATORS[fgen_name](**conf)

        # Check if required MS data is available
        missing_ms_data = fgen.required_ms_data - available_ms_data
        if missing_ms_data:
            logger.warning(
                f"Skipping feature generator {fgen_name} because required MS data is missing: "
                f"{missing_ms_data}. Ensure that the required MS data is present in the input "
                "files or disable the feature generator."
            )
            continue

        # Add features
        fgen.add_features(psm_list)
        logger.debug(f"Adding features from {fgen_name}: {set(fgen.feature_names)}")
        feature_names[fgen_name] = set(fgen.feature_names)

    # Filter out psms that do not have all added features
    all_feature_names = {f for fgen in feature_names.values() for f in fgen}
    psms_with_features = [
        (set(psm.rescoring_features.keys()) == all_feature_names) for psm in psm_list
    ]

    if psms_with_features.count(False) > 0:
        removed_psms = psm_list[[not psm for psm in psms_with_features]]
        missing_features = {
            feature_name
            for psm in removed_psms
            for feature_name in all_feature_names - set(psm.rescoring_features.keys())
        }
        logger.warning(
            f"Removed {psms_with_features.count(False)} PSMs that were missing one or more "
            f"rescoring feature(s), {missing_features}."
        )
    psm_list = psm_list[psms_with_features]

    if "mumble" in config["psm_generator"]:
        # Remove PSMs where matched_ions_pct drops 25% below the original hit
        psm_list = filter_mumble_psms(psm_list, threshold=0.75)
        # Currently replace the score with the hyperscore for Mumble
        # psm_list["score"] = [ft["hyperscore"] for ft in psm_list["rescoring_features"]] # TODO: This is a temporary fix

    # Write feature names to file
    _write_feature_names(feature_names, output_file_root)

    # Rename PSMs to USIs if requested
    if config["rename_to_usi"]:
        logging.debug(f"Creating USIs for {len(psm_list)} PSMs")
        psm_list["spectrum_id"] = [psm.get_usi(as_url=False) for psm in psm_list]

    # If no rescoring engine is specified or DEBUG, write PSMs and features to PIN file
    if not config["rescoring_engine"] or config["log_level"] == "debug":
        logger.info(f"Writing added features to PIN file: {output_file_root}.psms.pin")
        psm_utils.io.write_file(
            psm_list,
            output_file_root + ".pin",
            filetype="percolator",
            feature_names=all_feature_names,
        )

    if not config["rescoring_engine"]:
        logger.info("No rescoring engine specified. Skipping rescoring.")
        return None

    # Rescore PSMs
    try:
        if "percolator" in config["rescoring_engine"]:
            percolator.rescore(
                psm_list,
                output_file_root=output_file_root,
                log_level=config["log_level"],
                processes=config["processes"],
                percolator_kwargs=config["rescoring_engine"]["percolator"],
            )
        elif "mokapot" in config["rescoring_engine"]:
            if "fasta_file" not in config["rescoring_engine"]["mokapot"]:
                config["rescoring_engine"]["mokapot"]["fasta_file"] = config["fasta_file"]
            if "protein_kwargs" in config["rescoring_engine"]["mokapot"]:
                protein_kwargs = config["rescoring_engine"]["mokapot"].pop("protein_kwargs")
            else:
                protein_kwargs = dict()

            mokapot.rescore(
                psm_list,
                output_file_root=output_file_root,
                protein_kwargs=protein_kwargs,
                **config["rescoring_engine"]["mokapot"],
            )
    except exceptions.RescoringError as e:
        # Write output
        logger.info(f"Writing intermediary output to {output_file_root}.psms.tsv...")
        psm_utils.io.write_file(psm_list, output_file_root + ".psms.tsv", filetype="tsv")

        # Reraise exception
        raise e

    # Post-rescoring processing
    if all(psm_list["pep"] == 1.0):
        psm_list = _fix_constant_pep(psm_list)
    psm_list = _filter_by_rank(psm_list, config["max_psm_rank_output"], False)
    psm_list = _calculate_confidence(psm_list)
    _ = _log_id_psms_after(psm_list, id_psms_before, max_rank=config["max_psm_rank_output"])

    # Write output
    logger.info(f"Writing output to {output_file_root}.psms.tsv...")
    psm_utils.io.write_file(psm_list, output_file_root + ".psms.tsv", filetype="tsv")

    if config["write_flashlfq"]:
        logger.info(f"Writing output to {output_file_root}.flashlfq.tsv...")
        psm_utils.io.write_file(
            psm_list,
            output_file_root + ".flashlfq.tsv",
            filetype="flashlfq",
            fdr_threshold=0.01,
            only_target=True,  # TODO: Make FDR threshold configurable
        )

    # Write report
    if config["write_report"]:
        try:
            generate.generate_report(
                output_file_root, psm_list=psm_list, feature_names=feature_names, use_txt_log=True
            )
        except exceptions.ReportGenerationError as e:
            logger.exception(e)


def _filter_by_rank(psm_list: PSMList, max_rank: int, lower_score_better: bool) -> PSMList:
    """Filter PSMs by rank."""
    psm_list.set_ranks(lower_score_better=lower_score_better)
    rank_filter = psm_list["rank"] <= max_rank
    logger.info(f"Removed {sum(~rank_filter)} PSMs with rank >= {max_rank}.")
    return psm_list[rank_filter]


def _write_feature_names(feature_names, output_file_root):
    """Write feature names to file."""
    with open(output_file_root + ".feature_names.tsv", "w") as f:
        f.write("feature_generator\tfeature_name\n")
        for fgen, fgen_features in feature_names.items():
            for feature in fgen_features:
                f.write(f"{fgen}\t{feature}\n")


def _log_id_psms_before(psm_list: PSMList, fdr: float = 0.01, max_rank: int = 1) -> int:
    """Log #PSMs identified before rescoring."""
    id_psms_before = (
        (psm_list["qvalue"] <= 0.01)
        & (psm_list["rank"] <= max_rank)
        & (~psm_list["is_decoy"])
        & ([metadata.get("original_psm", True) for metadata in psm_list["metadata"]])
    ).sum()
    logger.info(
        f"Found {id_psms_before} identified PSMs with rank <= {max_rank} at {fdr} FDR before "
        "rescoring."
    )
    return id_psms_before


def _log_id_psms_after(
    psm_list: PSMList, id_psms_before: int, fdr: float = 0.01, max_rank: int = 1
) -> int:
    """Log #PSMs identified after rescoring."""
    id_psms_after = (
        (psm_list["qvalue"] <= 0.01) & (psm_list["rank"] <= max_rank) & (~psm_list["is_decoy"])
    ).sum()
    diff = id_psms_after - id_psms_before
    diff_perc = diff / id_psms_before if id_psms_before > 0 else None

    diff_numbers = f"{diff} ({diff_perc:.2%})" if diff_perc is not None else str(diff)
    diff_word = "more" if diff > 0 else "less"
    logger.info(
        f"Identified {diff_numbers} {diff_word} PSMs with rank <= {max_rank} at {fdr} FDR after "
        "rescoring."
    )

    return id_psms_after


def _fix_constant_pep(psm_list: PSMList) -> PSMList:
    """Workaround for broken PEP calculation if best PSM is decoy."""
    logger.warning(
        "Attempting to fix constant PEP values by removing decoy PSMs that score higher than the "
        "best target PSM."
    )
    max_target_score = psm_list["score"][~psm_list["is_decoy"]].max()
    higher_scoring_decoys = psm_list["is_decoy"] & (psm_list["score"] > max_target_score)

    if not higher_scoring_decoys.any():
        logger.warning("No decoys scoring higher than the best target found. Skipping fix.")
    else:
        psm_list = psm_list[~higher_scoring_decoys]
        logger.warning(f"Removed {higher_scoring_decoys.sum()} decoy PSMs.")

    return psm_list


def _calculate_confidence(psm_list: PSMList) -> PSMList:
    """
    Calculate scores, q-values, and PEPs for PSMs and peptides and add them to PSMList.
    """
    # Minimal conversion to LinearPsmDataset
    psm_df = psm_list.to_dataframe()
    psm_df = psm_df.reset_index(drop=True).reset_index()
    psm_df["peptide"] = (
        psm_df["peptidoform"].astype(str).str.replace(r"(/\d+$)", "", n=1, regex=True)
    )
    psm_df["is_target"] = ~psm_df["is_decoy"]
    lin_psm_data = LinearPsmDataset(
        psms=psm_df[["index", "peptide", "is_target"]],
        target_column="is_target",
        spectrum_columns="index",  # Use artificial index to allow multi-rank rescoring
        peptide_column="peptide",
    )

    # Recalculate confidence
    new_confidence = lin_psm_data.assign_confidence(
        scores=list(psm_list["score"])
    )  # explicity make it a list to avoid TypingError: Failed in nopython mode pipeline (step: nopython frontend) in mokapot

    # Add new confidence estimations to PSMList
    add_psm_confidence(psm_list, new_confidence)
    add_peptide_confidence(psm_list, new_confidence)

    return psm_list
