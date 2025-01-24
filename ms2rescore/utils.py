import logging
import os
from glob import glob
from pathlib import Path
from typing import Optional, Union
import numpy as np

from ms2rescore.exceptions import MS2RescoreConfigurationError
from ms2rescore_rs import is_supported_file_type
from psm_utils import PSMList

logger = logging.getLogger(__name__)


def infer_spectrum_path(
    configured_path: Union[str, Path, None],
    run_name: Optional[str] = None,
) -> Union[str, Path]:
    """
    Infer spectrum path from passed path and expected filename (e.g. from PSM file).

    Parameters
    ----------
    configured_path: str, Path, None
        User-defined path to spectrum file or directory containing spectrum file
    run_name : str, optional
        MS run name (stem of spectrum filename), e.g., as expected from PSM file.

    """
    # If no spectrum path configured, use expected run_name in default dir
    if not configured_path:
        if run_name:
            resolved_path = os.path.join(".", run_name)
        else:
            raise MS2RescoreConfigurationError(
                "Could not resolve spectrum file name: No spectrum path configured "
                "and no run name in PSM file found."
            )

    else:
        is_bruker_dir = configured_path.endswith(".d") or _is_minitdf(configured_path)

        # If passed path is directory (that is not Bruker raw), join with run name
        if os.path.isdir(configured_path) and not is_bruker_dir:
            if run_name:
                resolved_path = os.path.join(configured_path, run_name)
            else:
                raise MS2RescoreConfigurationError(
                    "Could not resolve spectrum file name: Spectrum path is directory "
                    "but no run name in PSM file found."
                )

        # If passed path is file, use that, but warn if basename doesn't match expected
        elif os.path.isfile(configured_path) or (os.path.isdir(configured_path) and is_bruker_dir):
            if run_name and Path(configured_path).stem != Path(run_name).stem:
                logger.warning(
                    "Passed spectrum path (`%s`) does not match run name found in PSM "
                    "file (`%s`). Continuing with passed spectrum path.",
                    configured_path,
                    run_name,
                )
            resolved_path = configured_path
        else:
            raise MS2RescoreConfigurationError(
                "Configured `spectrum_path` must be `None` or a path to an existing file "
                "or directory. If `None` or path to directory, spectrum run information "
                "should be present in the PSM file."
            )

    # Match with file extension if not in resolved_path yet
    if not is_supported_file_type(resolved_path) or not os.path.exists(resolved_path):
        for filename in glob(resolved_path + "*"):
            if is_supported_file_type(filename):
                resolved_path = filename
                break
        else:
            raise MS2RescoreConfigurationError(
                f"Resolved spectrum filename ('{resolved_path}') does not contain a supported "
                "file extension (mzML, MGF, or .d) and could not find any matching existing "
                "files."
            )

    return Path(resolved_path)


def _is_minitdf(spectrum_file: str) -> bool:
    """
    Check if the spectrum file is a Bruker miniTDF folder.

    A Bruker miniTDF folder has no fixed name, but contains files matching the patterns
    ``*ms2spectrum.bin`` and ``*ms2spectrum.parquet``.
    """
    files = set(Path(spectrum_file).glob("*ms2spectrum.bin"))
    files.update(Path(spectrum_file).glob("*ms2spectrum.parquet"))
    return len(files) >= 2


def filter_mumble_psms(psm_list: PSMList, threshold=1) -> PSMList:
    """
    Filter out mumble PSMs with `matched_ions_pct` lower than the original hit.

    Parameters
    ----------
    psm_list : PSMList
        List of PSMs to filter
    threshold : float, optional
        Threshold to lower the maximum matched_ions_pct of the original hit
    """
    # Extract relevant fields from the PSM list
    original_hit = np.array([metadata.get("original_psm") for metadata in psm_list["metadata"]])
    spectrum_indices = np.array([psm.spectrum_id for psm in psm_list])
    runs = np.array([psm.run for psm in psm_list])

    # Check if matched_ions_pct exists
    if "matched_ions_pct" not in psm_list[0].rescoring_features:
        return psm_list

    matched_ions = np.array([psm.rescoring_features["matched_ions_pct"] for psm in psm_list])

    # Create unique keys for each (run, spectrum_id)
    unique_keys = np.core.defchararray.add(runs.astype(str), spectrum_indices.astype(str))
    unique_keys_indices, inverse_indices = np.unique(unique_keys, return_inverse=True)

    # Initialize an array to store the `matched_ions_pct` of original hits per group
    original_matched_ions_pct = np.full(
        len(unique_keys_indices), -np.inf
    )  # Default to -inf for groups without original hits

    # Assign the `matched_ions_pct` of original hits to their groups
    np.maximum.at(
        original_matched_ions_pct, inverse_indices[original_hit], matched_ions[original_hit]
    )

    # lower the maximum with the threshold
    original_matched_ions_pct = original_matched_ions_pct * threshold

    # Broadcast the original `matched_ions_pct` back to all PSMs in each group
    original_matched_ions_for_all = original_matched_ions_pct[inverse_indices]

    # Determine the filtering condition
    keep = np.logical_or(
        original_hit,  # Always keep original hits
        matched_ions
        >= original_matched_ions_for_all,  # Keep hits with `matched_ions_pct` >= the original
    )

    # Filter PSMs
    logger.debug(f"Filtered out {len(psm_list) - np.sum(keep)} mumble PSMs.")
    return psm_list[keep]
