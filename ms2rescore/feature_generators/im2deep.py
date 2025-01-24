"""
IM2Deep ion mobility-based feature generator.

IM2Deep is a fully modification-aware peptide ion mobility predictor. It uses a deep convolutional
neural network to predict retention times based on the atomic composition of the (modified) amino
acid residues in the peptide. See
`github.com/compomics/IM2Deep <https://github.com/compomics/IM2Deep>`_ for more information.

"""

import contextlib
import logging
import os
from inspect import getfullargspec
from itertools import chain
from typing import List

import numpy as np
import pandas as pd
from im2deep.utils import im2ccs
from im2deep.im2deep import predict_ccs
from psm_utils import PSMList

from ms2rescore.feature_generators.base import FeatureGeneratorBase
from ms2rescore.parse_spectra import MSDataType

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logger = logging.getLogger(__name__)


class IM2DeepFeatureGenerator(FeatureGeneratorBase):
    """IM2Deep collision cross section feature generator."""

    required_ms_data = {MSDataType.ion_mobility}

    def __init__(
        self,
        *args,
        processes: int = 1,
        **kwargs,
    ):
        """
        Initialize the IM2DeepFeatureGenerator.

        Parameters
        ----------
        processes : int, optional
            Number of parallel processes to use for IM2Deep predictions. Default is 1.
        **kwargs : dict, optional
            Additional keyword arguments to `im2deep.predict_ccs`.

        """
        super().__init__(*args, **kwargs)

        self._verbose = logger.getEffectiveLevel() <= logging.DEBUG

        # Remove any kwargs that are not IM2Deep arguments
        self.im2deep_kwargs = kwargs or {}
        self.im2deep_kwargs = {
            k: v for k, v in self.im2deep_kwargs.items() if k in getfullargspec(predict_ccs).args
        }
        self.im2deep_kwargs["n_jobs"] = processes

    @property
    def feature_names(self) -> List[str]:
        return [
            "ccs_observed_im2deep",
            "ccs_predicted_im2deep",
            "ccs_error_im2deep",
            "abs_ccs_error_im2deep",
            "perc_ccs_error_im2deep",
        ]

    def add_features(self, psm_list: PSMList) -> None:
        """Add IM2Deep-derived features to PSMs"""
        logger.info("Adding IM2Deep-derived features to PSMs")

        # Get easy-access nested version of PSMlist
        psm_dict = psm_list.get_psm_dict()

        # Run IM2Deep for each spectrum file
        current_run = 1
        total_runs = sum(len(runs) for runs in psm_dict.values())

        for runs in psm_dict.values():
            # Reset IM2Deep predictor for each collection of runs
            for run, psms in runs.items():
                logger.info(
                    f"Running IM2Deep for PSMs from run ({current_run}/{total_runs}): `{run}`..."
                )

                # Disable wild logging to stdout by TensorFlow, unless in debug mode
                with (
                    contextlib.redirect_stdout(open(os.devnull, "w", encoding="utf-8"))
                    if not self._verbose
                    else contextlib.nullcontext()
                ):
                    # Make new PSM list for this run (chain PSMs per spectrum to flat list)
                    psm_list_run = PSMList(psm_list=list(chain.from_iterable(psms.values())))

                    logger.debug("Calibrating IM2Deep...")

                    # Convert ion mobility to CCS and calibrate CCS values
                    psm_list_run_df = psm_list_run.to_dataframe()
                    psm_list_run_df["charge"] = [
                        pep.precursor_charge for pep in psm_list_run_df["peptidoform"]
                    ]
                    psm_list_run_df["ccs_observed"] = im2ccs(
                        psm_list_run_df["ion_mobility"],
                        psm_list_run_df["precursor_mz"],
                        psm_list_run_df["charge"],
                    )

                    # Create dataframe with high confidence hits for calibration
                    cal_psm_df = self.make_calibration_df(psm_list_run_df)

                    # Make predictions with IM2Deep
                    logger.debug("Predicting CCS values...")
                    predictions = predict_ccs(
                        psm_list_run, cal_psm_df, write_output=False, **self.im2deep_kwargs
                    )

                    # Add features to PSMs
                    logger.debug("Adding features to PSMs...")
                    observations = psm_list_run_df["ccs_observed"]
                    ccs_diffs_run = np.abs(predictions - observations)
                    for i, psm in enumerate(psm_list_run):
                        psm["rescoring_features"].update(
                            {
                                "ccs_observed_im2deep": observations[i],
                                "ccs_predicted_im2deep": predictions[i],
                                "ccs_error_im2deep": ccs_diffs_run[i],
                                "abs_ccs_error_im2deep": np.abs(ccs_diffs_run[i]),
                                "perc_ccs_error_im2deep": np.abs(ccs_diffs_run[i])
                                / observations[i]
                                * 100,
                            }
                        )

                current_run += 1

    @staticmethod
    def make_calibration_df(psm_list_df: pd.DataFrame, threshold: float = 0.25) -> pd.DataFrame:
        """
        Make dataframe for calibration of IM2Deep predictions.

        Parameters
        ----------
        psm_list_df
            DataFrame with PSMs.
        threshold
            Percentage of highest scoring identified target PSMs to use for calibration,
            default 0.95.

        Returns
        -------
        pd.DataFrame
            DataFrame with high confidence hits for calibration.

        """
        identified_psms = psm_list_df[
            (psm_list_df["qvalue"] < 0.01)
            & (~psm_list_df["is_decoy"])
            & (psm_list_df["charge"] < 7)  # predictions do not go higher for IM2Deep
            & ([metadata.get("original_psm", True) for metadata in psm_list_df["metadata"]])
        ]
        calibration_psms = identified_psms[
            identified_psms["qvalue"] < identified_psms["qvalue"].quantile(1 - threshold)
        ]
        logger.debug(
            f"Number of high confidence hits for calculating shift: {len(calibration_psms)}"
        )
        return calibration_psms
