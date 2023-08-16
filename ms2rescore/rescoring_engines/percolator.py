import logging
import subprocess
from typing import Any, Dict, Optional

import numpy as np
import psm_utils

from ms2rescore.exceptions import MS2RescoreError

logger = logging.getLogger(__name__)


LOG_LEVEL_MAP = {
    "critical": 0,
    "error": 0,
    "warning": 0,
    "info": 1,
    "debug": 2,
}


def rescore(
    psm_list: psm_utils.PSMList,
    output_file_root: str = "ms2rescore",
    log_level: str = "info",
    processes: int = 1,
    percolator_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Rescore PSMs with Percolator.

    Parameters
    ----------
    psm_list
        PSMs to be rescored.
    output_file_root
        Root of output file names. Defaults to ``ms2rescore``.
    log_level
        Log level for Percolator. Defaults to ``info``.
    processes
        Number of processes to use. Defaults to 1.
    percolator_kwargs
        Additional keyword arguments for Percolator. Defaults to ``None``.

    """
    percolator_kwargs = {
        "results-psms": output_file_root + "_target_psms.pout",
        "decoy-results-psms": output_file_root + "_decoy_psms.pout",
        "results-peptides": output_file_root + "_target_peptides.pout",
        "decoy-results-peptides": output_file_root + "_decoy_peptides.pout",
        "results-proteins": output_file_root + "_target_proteins.pout",
        "decoy-results-proteins": output_file_root + "_decoy_proteins.pout",
        "weights": output_file_root + ".weights",
        "verbose": LOG_LEVEL_MAP[log_level],
        "num-threads": processes,
        "post-processing-tdc": True,
    }
    if percolator_kwargs:
        percolator_kwargs.update(percolator_kwargs)

    pin_filepath = f"{output_file_root}.pin"
    percolator_cmd = _construct_percolator_command(percolator_kwargs, pin_filepath)

    # Need to be able to link back to original PSMs, so reindex spectrum IDs, but copy PSM list
    # to avoid modifying original...
    # TODO: Better approach for this?
    psm_list_reindexed = psm_list.copy()
    psm_list_reindexed["spectrum_id"] = np.arange(len(psm_list_reindexed))

    _write_pin_file(psm_list_reindexed, pin_filepath)

    logger.debug(f"Running percolator command {' '.join(percolator_cmd)}")
    try:
        output = subprocess.run(
            percolator_cmd,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        logger.warn(f"Running Percolator resulted in an error:\n{output.stdout}")
        raise MS2RescoreError("Percolator error")

    logger.info(
        "Percolator output: \n" + _decode_string(output.stderr), extra={"highlighter": None}
    )

    _update_psm_scores(
        psm_list,
        percolator_kwargs["results-psms"],
        percolator_kwargs["decoy-results-psms"],
    )


def _update_psm_scores(psm_list: psm_utils.PSMList, target_pout: str, decoy_pout: str):
    """
    Update PSM scores with Percolator results.

    PSMs from the target and decoy pout files are mapped back by their collection, run,
    spectrum_id, and peptidoform.

    """
    target_psms = psm_utils.io.read_file(target_pout, filetype="percolator")
    decoy_psms = psm_utils.io.read_file(decoy_pout, filetype="percolator")
    psm_list_percolator = psm_utils.PSMList(psm_list=target_psms.psm_list + decoy_psms.psm_list)

    # Sort by reindexed spectrum_id so order matches original PSM list
    psm_list_percolator[np.argsort(psm_list_percolator["spectrum_id"])]

    if not len(psm_list) == len(psm_list_percolator):
        raise MS2RescoreError(
            f"Number of PSMs in original list ({len(psm_list)}) does not match number of PSMs in "
            f"Percolator output ({len(psm_list_percolator)})"
        )

    for original_psm, new_psm in zip(psm_list, psm_list_percolator):
        original_psm["score"] = new_psm["score"]
        original_psm["qvalue"] = new_psm["qvalue"]
        original_psm["pep"] = new_psm["pep"]


def _write_pin_file(psm_list: psm_utils.PSMList, filepath: str):
    """Write PIN file for rescoring."""
    logger.debug(f"Writing PIN file to {filepath}")
    psm_utils.io.write_file(
        psm_list,
        filename=filepath,
        filetype="percolator",
        style="pin",
        feature_names=psm_list[0].rescoring_features.keys(),
    )


def _construct_percolator_command(percolator_kwargs: Dict, pin_filepath: str):
    """Create Percolator command for given set of arguments and path to PIN file."""
    percolator_cmd = ["percolator"]
    for key, value in percolator_kwargs.items():
        if not isinstance(value, bool):
            percolator_cmd.append(f"--{key}")
            percolator_cmd.append(str(value))
            if key == "init-weights":
                percolator_cmd.append("--static")
        elif isinstance(value, bool) & value is False:
            continue
        else:
            percolator_cmd.append(f"--{key}")
    percolator_cmd.append(pin_filepath)
    return percolator_cmd


def _decode_string(encoded_string):
    for encoding in ["utf-8", "latin-1", "ascii", "iso-8859-15"]:
        try:
            decoded_string = encoded_string.decode(encoding)
            logger.debug(f"Decoded stderr with {encoding}")
            return decoded_string
        except UnicodeDecodeError:
            pass
    else:
        raise MS2RescoreError("Could not infer encoding of Percolator logs.")
