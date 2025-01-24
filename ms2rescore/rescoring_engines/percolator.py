"""
Percolator integration for MS²Rescore

Percolator was the first tool to introduce semi-supervised learning for PSM rescoring. It is
still widely used and has been integrated in many proteomics data analysis pipelines. This module
integrates with Percolator through its command line interface. Percolator must be installed
separately and the ``percolator`` command must be available in the PATH for this module to work.
See `github.com/percolator/percolator <https://github.com/percolator/percolator>`_	for
more information.

If you use Percolator through MS²Rescore, please cite:

.. epigraph::
    The M, MacCoss MJ, Noble WS, Käll L. Fast and Accurate Protein False Discovery Rates on
    Large-Scale Proteomics Data Sets with Percolator 3.0. *J Am Soc Mass Spectrom* (2016).
    `doi:10.1007/s13361-016-1460-7 <https://doi.org/10.1007/s13361-016-1460-7>`_

"""

import logging
import subprocess
from copy import deepcopy
from typing import Any, Dict, Optional

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
    fasta_file: Optional[str] = None,
    percolator_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Rescore PSMs with Percolator.

    Aside from updating the PSM ``score``, ``qvalue``, and ``pep`` values, the following output
    files are written:

        - Target PSMs: ``{output_file_root}.percolator.psms.pout``
        - Target peptides: ``{output_file_root}.percolator.peptides.pout``
        - Target proteins: ``{output_file_root}.percolator.proteins.pout``
        - Decoy PSMs: ``{output_file_root}.percolator.decoy.psms.pout``
        - Decoy peptides: ``{output_file_root}.percolator.decoy.peptides.pout``
        - Decoy proteins: ``{output_file_root}.percolator.decoy.proteins.pout``
        - Feature weights: ``{output_file_root}.percolator.weights.tsv``

    Percolator is run through its command line interface. Percolator must be installed separately
    and the ``percolator`` command must be available in the PATH for this module to work.

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
    fasta_file
        Path to FASTA file for protein inference. Defaults to ``None``.
    percolator_kwargs
        Additional keyword arguments for Percolator. Defaults to ``None``.

    """
    percolator_kwargs = {
        "results-psms": output_file_root + ".percolator.psms.pout",
        "decoy-results-psms": output_file_root + ".percolator.decoy.psms.pout",
        "results-peptides": output_file_root + ".percolator.peptides.pout",
        "decoy-results-peptides": output_file_root + ".percolator.decoy.peptides.pout",
        "results-proteins": output_file_root + ".percolator.proteins.pout",
        "decoy-results-proteins": output_file_root + ".percolator.decoy.proteins.pout",
        "weights": output_file_root + ".percolator.weights.tsv",
        "verbose": LOG_LEVEL_MAP[log_level],
        "num-threads": min(processes, 128),  # Higher values not supported by Percolator
        "post-processing-tdc": True,
    }
    if percolator_kwargs:
        percolator_kwargs.update(percolator_kwargs)

    if fasta_file:
        percolator_kwargs["picked-protein"] = fasta_file

    pin_filepath = f"{output_file_root}.pin"
    percolator_cmd = _construct_percolator_command(percolator_kwargs, pin_filepath)

    # Need to be able to link back to original PSMs, so reindex spectrum IDs, but copy PSM list
    # to avoid modifying original...
    # TODO: Better approach for this?

    psm_list_reindexed = deepcopy(psm_list)
    psm_list_reindexed.set_ranks()
    psm_list_reindexed["spectrum_id"] = [
        f"{psm.get_usi(as_url=False)}_{psm.rank}" for psm in psm_list_reindexed
    ]
    spectrum_id_index = {
        spectrum_id: index for index, spectrum_id in enumerate(psm_list_reindexed["spectrum_id"])
    }

    _write_pin_file(psm_list_reindexed, pin_filepath)

    logger.debug(f"Running percolator command {' '.join(percolator_cmd)}")
    try:
        output = subprocess.run(percolator_cmd, capture_output=True)
    except FileNotFoundError as e:
        if subprocess.getstatusoutput("percolator")[0] != 0:
            raise MS2RescoreError(
                "Could not run Percolator. Please ensure that the program is installed and "
                "available in your PATH. See "
                "https://ms2rescore.readthedocs.io/en/latest/installation/#installing-percolator "
                "for more information."
            ) from e
        else:
            logger.warn(f"Running Percolator resulted in an error:\n{output.stdout}")
            raise MS2RescoreError("Percolator error") from e
    except subprocess.CalledProcessError as e:
        logger.warn(f"Running Percolator resulted in an error:\n{output.stdout}")
        raise MS2RescoreError("Percolator error") from e

    logger.info(
        "Percolator output: \n" + _decode_string(output.stderr), extra={"highlighter": None}
    )

    _update_psm_scores(
        psm_list,
        percolator_kwargs["results-psms"],
        percolator_kwargs["decoy-results-psms"],
        spectrum_id_index,
    )


def _update_psm_scores(
    psm_list: psm_utils.PSMList, target_pout: str, decoy_pout: str, spectrum_id_index: list
):
    """
    Update PSM scores with Percolator results.

    PSMs from the target and decoy pout files are mapped back by their collection, run,
    spectrum_id, and peptidoform.

    """
    target_psms = psm_utils.io.read_file(target_pout, filetype="percolator")
    decoy_psms = psm_utils.io.read_file(decoy_pout, filetype="percolator")
    psm_list_percolator = psm_utils.PSMList(psm_list=target_psms.psm_list + decoy_psms.psm_list)

    # Sort by reindexed spectrum_id so order matches original PSM list
    psm_list_percolator = sorted(
        psm_list_percolator, key=lambda psm: spectrum_id_index[psm["spectrum_id"]]
    )

    if not len(psm_list) == len(psm_list_percolator):
        raise MS2RescoreError(
            f"Number of PSMs in original list ({len(psm_list)}) does not match number of PSMs in "
            f"Percolator output ({len(psm_list_percolator)})"
        )

    for original_psm, new_psm in zip(psm_list, psm_list_percolator):
        original_psm["score"] = new_psm["score"]
        original_psm["qvalue"] = new_psm["qvalue"]
        original_psm["pep"] = new_psm["pep"]

    psm_list.set_ranks(lower_score_better=False)


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
