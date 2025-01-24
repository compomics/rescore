"""
DeepLC retention time-based feature generator.

DeepLC is a fully modification-aware peptide retention time predictor. It uses a deep convolutional
neural network to predict retention times based on the atomic composition of the (modified) amino
acid residues in the peptide. See
`github.com/compomics/deeplc <https://github.com/compomics/deeplc>`_ for more information.

If you use DeepLC through MS²Rescore, please cite:

.. epigraph::
    Bouwmeester, R., Gabriels, R., Hulstaert, N. et al. DeepLC can predict retention times for
    peptides that carry unknown modifications. *Nat Methods* 18, 1363-1369 (2021).
    `doi:10.1038/s41592-021-01301-5 <https://doi.org/10.1038/s41592-021-01301-5>`_

"""

import contextlib
import logging
import os
from collections import defaultdict
from inspect import getfullargspec
from itertools import chain
from typing import List, Union

import numpy as np
from psm_utils import PSMList

from ms2rescore.feature_generators.base import FeatureGeneratorBase
from ms2rescore.parse_spectra import MSDataType

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logger = logging.getLogger(__name__)


class DeepLCFeatureGenerator(FeatureGeneratorBase):
    """DeepLC retention time-based feature generator."""

    required_ms_data = {MSDataType.retention_time}

    def __init__(
        self,
        *args,
        lower_score_is_better: bool = False,
        calibration_set_size: Union[int, float, None] = None,
        calibration_set: Union[str, None] = None,
        processes: int = 1,
        **kwargs,
    ) -> None:
        """
        Generate DeepLC-based features for rescoring.

        DeepLC retraining is on by default. Add ``deeplc_retrain: False`` as a keyword argument to
        disable retraining.

        Parameters
        ----------
        lower_score_is_better
            Whether a lower PSM score denotes a better matching PSM. Default: False
        calibration_set_size: int or float
            Amount of best PSMs to use for DeepLC calibration. If this value is lower
            than the number of available PSMs, all PSMs will be used. (default: 0.15)
        processes: {int, None}
            Number of processes to use in DeepLC. Defaults to 1.
        kwargs: dict
            Additional keyword arguments are passed to DeepLC.

        Attributes
        ----------
        feature_names: list[str]
            Names of the features that will be added to the PSMs.

        """
        super().__init__(*args, **kwargs)

        self.lower_psm_score_better = lower_score_is_better
        self.calibration_set_size = calibration_set_size
        self.calibration_set = calibration_set
        self.processes = processes
        self.deeplc_kwargs = kwargs or {}

        self._verbose = logger.getEffectiveLevel() <= logging.DEBUG

        # Lazy-load DeepLC
        from deeplc import DeepLC

        self.DeepLC = DeepLC

        # Remove any kwargs that are not DeepLC arguments
        self.deeplc_kwargs = {
            k: v for k, v in self.deeplc_kwargs.items() if k in getfullargspec(DeepLC).args
        }
        self.deeplc_kwargs.update({"config_file": None})

        # Set default DeepLC arguments
        if "deeplc_retrain" not in self.deeplc_kwargs:
            self.deeplc_kwargs["deeplc_retrain"] = False

        self.deeplc_predictor = None
        if "path_model" in self.deeplc_kwargs:
            self.user_model = self.deeplc_kwargs.pop("path_model")
            logging.debug(f"Using user-provided DeepLC model {self.user_model}.")
        else:
            self.user_model = None

    @property
    def feature_names(self) -> List[str]:
        return [
            "observed_retention_time",
            "predicted_retention_time",
            "rt_diff",
            "observed_retention_time_best",
            "predicted_retention_time_best",
            "rt_diff_best",
        ]

    def add_features(self, psm_list: PSMList) -> None:
        """Add DeepLC-derived features to PSMs."""

        logger.info("Adding DeepLC-derived features to PSMs.")

        # Get easy-access nested version of PSMList
        psm_dict = psm_list.get_psm_dict()

        # Run DeepLC for each spectrum file
        current_run = 1
        total_runs = sum(len(runs) for runs in psm_dict.values())
        for runs in psm_dict.values():
            # Reset DeepLC predictor for each collection of runs
            self.deeplc_predictor = None
            self.selected_model = None
            for run, psms in runs.items():
                peptide_rt_diff_dict = defaultdict(
                    lambda: {
                        "observed_retention_time_best": np.Inf,
                        "predicted_retention_time_best": np.Inf,
                        "rt_diff_best": np.Inf,
                    }
                )
                logger.info(
                    f"Running DeepLC for PSMs from run ({current_run}/{total_runs}): `{run}`..."
                )

                # Disable wild logging to stdout by Tensorflow, unless in debug mode
                with (
                    contextlib.redirect_stdout(open(os.devnull, "w", encoding="utf-8"))
                    if not self._verbose
                    else contextlib.nullcontext()
                ):
                    # Make new PSM list for this run (chain PSMs per spectrum to flat list)
                    psm_list_run = PSMList(psm_list=list(chain.from_iterable(psms.values())))
                    psm_list_calibration = self._get_calibration_psms(psm_list_run)
                    logger.debug(f"Calibrating DeepLC with {len(psm_list_calibration)} PSMs...")
                    self.deeplc_predictor = self.DeepLC(
                        n_jobs=self.processes,
                        verbose=self._verbose,
                        path_model=self.selected_model or self.user_model,
                        **self.deeplc_kwargs,
                    )
                    self.deeplc_predictor.calibrate_preds(psm_list_calibration)
                    # Still calibrate for each run, but do not try out all model options.
                    # Just use model that was selected based on first run
                    if not self.selected_model:
                        self.selected_model = list(self.deeplc_predictor.model.keys())
                        self.deeplc_kwargs["deeplc_retrain"] = False
                        logger.debug(
                            f"Selected DeepLC model {self.selected_model} based on "
                            "calibration of first run. Using this model (after new "
                            "calibrations) for the remaining runs."
                        )

                    logger.debug("Predicting retention times...")
                    predictions = np.array(self.deeplc_predictor.make_preds(psm_list_run))
                    observations = psm_list_run["retention_time"]
                    rt_diffs_run = np.abs(predictions - observations)

                    logger.debug("Adding features to PSMs...")
                    for i, psm in enumerate(psm_list_run):
                        psm["rescoring_features"].update(
                            {
                                "observed_retention_time": observations[i],
                                "predicted_retention_time": predictions[i],
                                "rt_diff": rt_diffs_run[i],
                            }
                        )
                        peptide = psm.peptidoform.proforma.split("\\")[0]  # remove charge
                        if peptide_rt_diff_dict[peptide]["rt_diff_best"] > rt_diffs_run[i]:
                            peptide_rt_diff_dict[peptide] = {
                                "observed_retention_time_best": observations[i],
                                "predicted_retention_time_best": predictions[i],
                                "rt_diff_best": rt_diffs_run[i],
                            }
                    for psm in psm_list_run:
                        psm["rescoring_features"].update(
                            peptide_rt_diff_dict[psm.peptidoform.proforma.split("\\")[0]]
                        )
                current_run += 1

    def _get_calibration_psms(self, psm_list: PSMList):
        """Get N best scoring target PSMs for calibration."""
        psm_list_targets = psm_list[
            ~psm_list["is_decoy"]
            & [metadata.get("original_psm", True) for metadata in psm_list["metadata"]]
        ]
        if self.calibration_set_size:
            n_psms = self._get_number_of_calibration_psms(psm_list_targets)
            indices = np.argsort(psm_list_targets["score"])
            indices = indices[:n_psms] if self.lower_psm_score_better else indices[-n_psms:]
            return psm_list_targets[indices]
        else:
            identified_psms = psm_list_targets[psm_list_targets["qvalue"] <= 0.01]
            if len(identified_psms) == 0:
                raise ValueError(
                    "No target PSMs with q-value <= 0.01 found. Please set calibration set size for calibrating deeplc."
                )
            elif (len(identified_psms) < 500) & (self.deeplc_kwargs["deeplc_retrain"]):
                logger.warning(
                    " Less than 500 target PSMs with q-value <= 0.01 found for retraining. Consider turning of deeplc_retrain, as this is likely not enough data for retraining."
                )
            return identified_psms

    def _get_number_of_calibration_psms(self, psm_list):
        """Get number of calibration PSMs given `calibration_set_size` and total number of PSMs."""
        if isinstance(self.calibration_set_size, float):
            if not 0 < self.calibration_set_size <= 1:
                raise ValueError(
                    "If `calibration_set_size` is a float, it cannot be smaller than "
                    "or equal to 0 or larger than 1."
                )
            else:
                num_calibration_psms = round(len(psm_list) * self.calibration_set_size)
        elif isinstance(self.calibration_set_size, int):
            if self.calibration_set_size > len(psm_list):
                logger.warning(
                    f"Requested number of calibration PSMs ({self.calibration_set_size}"
                    f") is larger than total number of PSMs ({len(psm_list)}). Using "
                    "all PSMs for calibration."
                )
                num_calibration_psms = len(psm_list)
            else:
                num_calibration_psms = self.calibration_set_size
        else:
            raise TypeError(
                "Expected float or int for `calibration_set_size`. Got "
                f"{type(self.calibration_set_size)} instead. "
            )
        logger.debug(f"Using {num_calibration_psms} PSMs for calibration")
        return num_calibration_psms
