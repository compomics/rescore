import logging
import re
from typing import Dict, Optional, Union

import numpy as np
import psm_utils.io
from mumble import PSMHandler
from psm_utils import PSMList

from ms2rescore.exceptions import MS2RescoreConfigurationError

logger = logging.getLogger(__name__)


def parse_psms(config: Dict, psm_list: Union[PSMList, None]) -> PSMList:
    """
    Parse PSMs and prepare for rescoring.

    Parameters
    ----------
    config
        Dictionary containing general ms2rescore configuration (everything under ``ms2rescore``
        top-level key).
    psm_list
        PSMList object containing PSMs. If None, PSMs will be read from ``psm_file``.

    """
    # Read PSMs
    try:
        psm_list = _read_psms(config, psm_list)
    except psm_utils.io.PSMUtilsIOException:
        raise MS2RescoreConfigurationError(
            "Error occurred while reading PSMs. Please check the 'psm_file' and "
            "'psm_file_type' settings. See "
            "https://ms2rescore.readthedocs.io/en/latest/userguide/input-files/"
            " for more information."
        )

    # Filter by PSM rank
    psm_list.set_ranks(config["lower_score_is_better"])
    rank_filter = psm_list["rank"] <= config["max_psm_rank_input"]
    psm_list = psm_list[rank_filter]
    logger.info(f"Removed {sum(~rank_filter)} PSMs with rank >= {config['max_psm_rank_input']}.")

    # Remove invalid AAs, find decoys, calculate q-values
    psm_list = _remove_invalid_aa(psm_list)
    _find_decoys(psm_list, config["id_decoy_pattern"])
    _calculate_qvalues(psm_list, config["lower_score_is_better"])
    if config["psm_id_rt_pattern"] or config["psm_id_im_pattern"]:
        logger.debug("Parsing retention time and/or ion mobility from PSM identifier...")
        _parse_values_from_spectrum_id(
            psm_list, config["psm_id_rt_pattern"], config["psm_id_im_pattern"]
        )

    # Store scoring values for comparison later
    for psm in psm_list:
        psm.provenance_data.update(
            {
                "before_rescoring_score": psm.score,
                "before_rescoring_qvalue": psm.qvalue,
                "before_rescoring_pep": psm.pep,
                "before_rescoring_rank": psm.rank,
            }
        )

    logger.debug("Parsing modifications...")
    modifications_found = set(
        [
            re.search(r"\[([^\[\]]*)\]", x.proforma).group(1)
            for x in psm_list["peptidoform"]
            if "[" in x.proforma
        ]
    )
    logger.debug(f"Found modifications: {modifications_found}")
    non_mapped_modifications = modifications_found - set(config["modification_mapping"].keys())
    if non_mapped_modifications:
        logger.warning(
            f"Non-mapped modifications found: {non_mapped_modifications}\n"
            "This can be ignored if they are Unimod modification labels."
        )
    psm_list.rename_modifications(config["modification_mapping"])
    psm_list.add_fixed_modifications(config["fixed_modifications"])
    psm_list.apply_fixed_modifications()

    if config["psm_id_pattern"]:
        pattern = re.compile(config["psm_id_pattern"])
        logger.debug("Applying 'psm_id_pattern'...")
        logger.debug(
            f"Parsing '{psm_list[0].spectrum_id}' to '{_match_psm_ids(psm_list[0].spectrum_id, pattern)}'"
        )
        new_ids = [_match_psm_ids(old_id, pattern) for old_id in psm_list["spectrum_id"]]
        psm_list["spectrum_id"] = new_ids

    # Addition of Modifications for mass shifts in the PSMs with Mumble
    if "mumble" in config["psm_generator"]:
        logger.debug("Applying modifications for mass shifts using Mumble...")

        # set inlcude original psm to True and include decoy psm to true
        if (
            "include_original_psm" not in config["psm_generator"]["mumble"]
            or not config["psm_generator"]["mumble"]
        ):
            config["psm_generator"]["mumble"]["include_original_psm"] = True
        if (
            "include_decoy_psm" not in config["psm_generator"]["mumble"]
            or not config["psm_generator"]["mumble"]
        ):
            config["psm_generator"]["mumble"]["include_decoy_psm"] = True

        mumble_config = config["psm_generator"]["mumble"]

        # Check if psm_list[0].rescoring_features is empty or not
        if psm_list[0].rescoring_features:
            logger.debug("Removing psm_file rescoring features from PSMs...")
            # psm_list.remove_rescoring_features() # TODO add this to psm_utils
            for psm in psm_list:
                psm.rescoring_features = {}

        psm_handler = PSMHandler(
            **mumble_config,
        )
        psm_list = psm_handler.add_modified_psms(psm_list)

    return psm_list


def _read_psms(config, psm_list):
    if isinstance(psm_list, PSMList):
        return psm_list
    else:
        total_files = len(config["psm_file"])
        psm_list = []
        for current_file, psm_file in enumerate(config["psm_file"]):
            logger.info(
                f"Reading PSMs from PSM file ({current_file+1}/{total_files}): '{psm_file}'..."
            )
            psm_list.extend(
                psm_utils.io.read_file(
                    psm_file,
                    filetype=config["psm_file_type"],
                    show_progressbar=True,
                    **config["psm_reader_kwargs"],
                )
            )
            logger.debug(f"Read {len(psm_list)} PSMs from '{psm_file}'.")

        return PSMList(psm_list=psm_list)


def _find_decoys(psm_list: PSMList, id_decoy_pattern: Optional[str] = None):
    """Find decoys in PSMs, log amount, and raise error if none found."""
    logger.debug("Finding decoys...")
    if id_decoy_pattern:
        psm_list.find_decoys(id_decoy_pattern)

    n_psms = len(psm_list)
    percent_decoys = sum(psm_list["is_decoy"]) / n_psms * 100
    logger.info(f"Found {n_psms} PSMs, of which {percent_decoys:.2f}% are decoys.")

    if not any(psm_list["is_decoy"]):
        raise MS2RescoreConfigurationError(
            "No decoy PSMs found. Please check if decoys are present in the PSM file and that "
            "the 'id_decoy_pattern' option is correct. See "
            "https://ms2rescore.readthedocs.io/en/latest/userguide/configuration/#selecting-decoy-psms"
            " for more information."
        )


def _calculate_qvalues(psm_list: PSMList, lower_score_is_better: bool):
    """Calculate q-values for PSMs if not present."""
    # Calculate q-values if not present
    if None in psm_list["qvalue"]:
        logger.debug("Recalculating q-values...")
        psm_list.calculate_qvalues(reverse=not lower_score_is_better)


def _match_psm_ids(old_id, regex_pattern):
    """Match PSM IDs to regex pattern or raise Exception if no match present."""
    match = re.search(regex_pattern, str(old_id))
    try:
        return match[1]
    except (TypeError, IndexError):
        raise MS2RescoreConfigurationError(
            f"'psm_id_pattern' could not be extracted from PSM spectrum IDs (i.e. {old_id})."
            " Ensure that the regex contains a capturing group?"
        )


def _parse_values_from_spectrum_id(
    psm_list: PSMList,
    psm_id_rt_pattern: Optional[str] = None,
    psm_id_im_pattern: Optional[str] = None,
):
    """Parse retention time and or ion mobility values from the spectrum_id."""
    for pattern, label, key in zip(
        [psm_id_rt_pattern, psm_id_im_pattern],
        ["retention time", "ion mobility"],
        ["retention_time", "ion_mobility"],
    ):
        if pattern:
            logger.debug(f"Parsing {label} from spectrum_id with regex pattern " f"{pattern}")
            try:
                pattern = re.compile(pattern)
                psm_list[key] = [
                    float(pattern.search(psm.spectrum_id).group(1)) for psm in psm_list
                ]
            except AttributeError:
                raise MS2RescoreConfigurationError(
                    f"Could not parse {label} from spectrum_id with the "
                    f"{pattern} regex pattern. "
                    f"Example spectrum_id: '{psm_list[0].spectrum_id}'\n. "
                    f"Please make sure the {label} key is present in the spectrum_id "
                    "and the value is in a capturing group or disable the relevant feature generator."
                )


def _remove_invalid_aa(psm_list: PSMList) -> PSMList:
    """Remove PSMs with invalid amino acids."""
    invalid_psms = np.array(
        [any(aa in "BJOUXZ" for aa in psm.peptidoform.sequence) for psm in psm_list]
    )

    if any(invalid_psms):
        logger.warning(f"Removed {sum(invalid_psms)} PSMs with invalid amino acids.")
        return psm_list[~invalid_psms]
    else:
        logger.debug("No PSMs with invalid amino acids found.")
        return psm_list
