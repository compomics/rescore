"""
``ionmob`` collisional cross section (CCS)-based feature generator.

``ionmob`` is a predictor for peptide collisional cross sections (CCS), as measured in ion mobility
devices, such as the Bruker timsTOF instruments. More info can be found on the
`ionmob GitHub page <https://github.com/theGreatHerrLebert/ionmob>`_.

If you use ``ionmob`` in your work, please cite the following publication:

.. epigraph::
    Teschner, D. et al. Ionmob: a Python package for prediction of peptide collisional
    cross-section values. *Bioinformatics* 39, btad486 (2023).
    `doi:10.1093/bioinformatics/btad486 <https://doi.org/10.1093/bioinformatics/btad486>`_

"""

import logging
from itertools import chain
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import tensorflow as tf
from psm_utils import Peptidoform, PSMList

from ms2rescore.feature_generators.base import FeatureGeneratorBase, FeatureGeneratorException
from ms2rescore.parse_spectra import MSDataType

try:
    from ionmob import __file__ as ionmob_file
    from ionmob.preprocess.data import to_tf_dataset_inference
    from ionmob.utilities.chemistry import (
        VARIANT_DICT,
        calculate_mz,
        reduced_mobility_to_ccs,
    )
    from ionmob.utilities.tokenization import tokenizer_from_json
    from ionmob.utilities.utility import get_ccs_shift
except ImportError:
    IONMOB_INSTALLED = False
else:
    IONMOB_INSTALLED = True

logger = logging.getLogger(__name__)

if IONMOB_INSTALLED:
    IONMOB_DIR = Path(ionmob_file).parent
    DEFAULT_MODELS_IONMOB = {
        Path("pretrained_models/DeepTwoMerModel"),
        Path("pretrained_models/GRUPredictor"),
        Path("pretrained_models/SqrtModel"),
    }
    DEFAULT_MODELS_DICT = {
        mod_path.stem: IONMOB_DIR / mod_path for mod_path in DEFAULT_MODELS_IONMOB
    }
    DEFAULT_TOKENIZER = IONMOB_DIR / "pretrained_models/tokenizers/tokenizer.json"
    DEFAULT_REFERENCE_DATASET = IONMOB_DIR / "example_data/Tenzer_unimod.parquet"


class IonMobFeatureGenerator(FeatureGeneratorBase):
    """Ionmob collisional cross section (CCS)-based feature generator."""

    required_ms_data = {MSDataType.ion_mobility}

    def __init__(
        self,
        *args,
        ionmob_model: str = "GRUPredictor",
        reference_dataset: Optional[str] = None,
        tokenizer: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Ionmob collisional cross section (CCS)-based feature generator.

        Parameters
        ----------
        *args
            Additional arguments passed to the base class.
        ionmob_model
            Path to a trained Ionmob model or one of the default models (``DeepTwoMerModel``,
            ``GRUPredictor``, or ``SqrtModel``). Default: ``GRUPredictor``.
        reference_dataset
            Path to a reference dataset for CCS shift calculation. Uses the default reference
            dataset if not specified.
        tokenizer
            Path to a tokenizer or one of the default tokenizers. Uses the default tokenizer if
            not specified.
        **kwargs
            Additional keyword arguments passed to the base class.

        """
        super().__init__(*args, **kwargs)

        # Check if Ionmob could be imported
        if not IONMOB_INSTALLED:
            raise ImportError(
                "Ionmob not installed. Please install Ionmob to use this feature generator."
            )

        # Get model from file or one of the default models
        if Path(ionmob_model).is_file():
            self.ionmob_model = tf.keras.models.load_model(ionmob_model)
        elif ionmob_model in DEFAULT_MODELS_DICT:
            self.ionmob_model = tf.keras.models.load_model(
                DEFAULT_MODELS_DICT[ionmob_model].as_posix()
            )
        else:
            raise IonmobException(
                f"Invalid Ionmob model: {ionmob_model}. Should be path to a model file or one of "
                f"the default models: {DEFAULT_MODELS_DICT.keys()}."
            )
        self.reference_dataset = pd.read_parquet(reference_dataset or DEFAULT_REFERENCE_DATASET)
        self.tokenizer = tokenizer_from_json(tokenizer or DEFAULT_TOKENIZER)

        self._verbose = logger.getEffectiveLevel() <= logging.DEBUG

    @property
    def feature_names(self):
        return [
            "ccs_predicted",
            "ccs_observed",
            "ccs_error",
            "abs_ccs_error",
            "perc_ccs_error",
        ]

    @property
    def allowed_modifications(self):
        """Return a list of modifications that are allowed in ionmob."""
        return [token for aa_tokens in VARIANT_DICT.values() for token in aa_tokens]

    def add_features(self, psm_list: PSMList) -> None:
        """
        Add Ionmob-derived features to PSMs.

        Parameters
        ----------
        psm_list
            PSMs to add features to.

        """
        logger.info("Adding Ionmob-derived features to PSMs.")
        psm_dict = psm_list.get_psm_dict()
        current_run = 1
        total_runs = sum(len(runs) for runs in psm_dict.values())

        for runs in psm_dict.values():
            for run, psms in runs.items():
                logger.info(
                    f"Running Ionmob for PSMs from run ({current_run}/{total_runs}): `{run}`..."
                )

                psm_list_run = PSMList(psm_list=list(chain.from_iterable(psms.values())))
                psm_list_run_df = psm_list_run.to_dataframe()

                # prepare data frames for CCS prediction
                psm_list_run_df["charge"] = [
                    peptidoform.precursor_charge for peptidoform in psm_list_run_df["peptidoform"]
                ]
                psm_list_run_df = psm_list_run_df[
                    psm_list_run_df["charge"] < 5
                ]  # predictions do not go higher for ionmob

                psm_list_run_df["sequence-tokenized"] = psm_list_run_df.apply(
                    lambda x: self.tokenize_peptidoform(x["peptidoform"]), axis=1
                )
                psm_list_run_df = psm_list_run_df[
                    psm_list_run_df.apply(
                        lambda x: self._is_valid_tokenized_sequence(x["sequence-tokenized"]),
                        axis=1,
                    )
                ]

                # TODO: Use observed m/z?
                psm_list_run_df["mz"] = psm_list_run_df.apply(
                    lambda x: calculate_mz(x["sequence-tokenized"], x["charge"]), axis=1
                )  # use precursor m/z from PSMs?

                psm_list_run_df["ccs_observed"] = psm_list_run_df.apply(
                    lambda x: reduced_mobility_to_ccs(x["ion_mobility"], x["mz"], x["charge"]),
                    axis=1,
                )
                # calibrate CCS values
                shift_factor = self.calculate_ccs_shift(psm_list_run_df)
                psm_list_run_df["ccs_observed"] + shift_factor

                # predict CCS values
                tf_ds = to_tf_dataset_inference(
                    psm_list_run_df["mz"],
                    psm_list_run_df["charge"],
                    psm_list_run_df["sequence-tokenized"],
                    self.tokenizer,
                )

                psm_list_run_df["ccs_predicted"], _ = self.ionmob_model.predict(tf_ds)

                # calculate CCS features
                ccs_features = self._calculate_features(psm_list_run_df)

                # add CCS features to PSMs
                for psm in psm_list_run:
                    try:
                        psm["rescoring_features"].update(ccs_features[psm.spectrum_id])
                    except KeyError:
                        psm["rescoring_features"].update({})
                current_run += 1

    def _calculate_features(self, feature_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Get CCS features for PSMs."""
        ccs_features = {}
        for row in feature_df.itertuples():
            ccs_features[row.spectrum_id] = {
                "ccs_predicted": row.ccs_predicted,
                "ccs_observed": row.ccs_observed,
                "ccs_error": row.ccs_observed - row.ccs_predicted,
                "abs_ccs_error": abs(row.ccs_observed - row.ccs_predicted),
                "perc_ccs_error": ((abs(row.ccs_observed - row.ccs_predicted)) / row.ccs_observed)
                * 100,
            }
        return ccs_features

    @staticmethod
    def tokenize_peptidoform(peptidoform: Peptidoform) -> list:
        """Tokenize proforma sequence and add modifications."""
        tokenized_seq = []

        if peptidoform.properties["n_term"]:
            tokenized_seq.append(
                f"<START>[UNIMOD:{peptidoform.properties['n_term'][0].definition['id']}]"
            )
        else:
            tokenized_seq.append("<START>")

        for amino_acid, modification in peptidoform.parsed_sequence:
            tokenized_seq.append(amino_acid)
            if modification:
                tokenized_seq[-1] = (
                    tokenized_seq[-1] + f"[UNIMOD:{modification[0].definition['id']}]"
                )

        if peptidoform.properties["c_term"]:
            pass  # provide if c-term mods are supported
        else:
            tokenized_seq.append("<END>")

        return tokenized_seq

    def calculate_ccs_shift(self, psm_dataframe: pd.DataFrame) -> float:
        """
        Apply CCS shift to CCS values.

        Parameters
        ----------
        psm_dataframe
            Dataframe with PSMs as returned by :py:meth:`psm_utils.PSMList.to_dataframe`.

        """
        df = psm_dataframe.copy()
        df.rename({"ccs_observed": "ccs"}, axis=1, inplace=True)
        high_conf_hits = list(df["spectrum_id"][df["score"].rank(pct=True) > 0.95])
        logger.debug(
            f"Number of high confidence hits for calculating shift: {len(high_conf_hits)}"
        )

        shift_factor = get_ccs_shift(
            df[["charge", "sequence-tokenized", "ccs"]][df["spectrum_id"].isin(high_conf_hits)],
            self.reference_dataset,
        )

        logger.debug(f"CCS shift factor: {shift_factor}")

        return shift_factor

    def _is_valid_tokenized_sequence(self, tokenized_seq):
        """
        Check if peptide sequence contains invalid tokens.

        Parameters
        ----------
        tokenized_seq
            Tokenized peptide sequence.

        Returns
        -------
        bool
            False if invalid tokens are present, True otherwise.

        """
        for token in tokenized_seq:
            if token not in self.allowed_modifications:
                logger.debug(f"Invalid modification found: {token}")
                return False
        return True


class IonmobException(FeatureGeneratorException):
    """Exception raised by Ionmob feature generator."""

    pass
