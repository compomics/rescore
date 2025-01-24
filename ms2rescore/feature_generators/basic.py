"""Generate basic features that can be extracted from any PSM list."""

import logging
from typing import Dict, Iterable, List, Tuple

import numpy as np
from psm_utils import PSMList

from ms2rescore.feature_generators.base import FeatureGeneratorBase

logger = logging.getLogger(__name__)


class BasicFeatureGenerator(FeatureGeneratorBase):
    def __init__(self, *args, **kwargs) -> None:
        """
        Generate basic features that can be extracted from any PSM list, including search engine
        score, charge state, and MS1 error.

        Parameters
        ----------
        *args
            Positional arguments passed to the base class.
        **kwargs
            Keyword arguments passed to the base class.

        Attributes
        ----------
        feature_names: list[str]
            Names of the features that will be added to the PSMs.

        """
        super().__init__(*args, **kwargs)
        self._feature_names = None

    @property
    def feature_names(self) -> List[str]:
        if self._feature_names is None:
            raise ValueError("Feature names have not been set yet. First run `add_features`.")
        return self._feature_names

    def add_features(self, psm_list: PSMList) -> None:
        """
        Add basic features to a PSM list.

        Parameters
        ----------
        psm_list
            PSM list to add features to.

        """
        logger.info("Adding basic features to PSMs.")

        self._feature_names = []  # Reset feature names

        charge_states = np.array([psm.peptidoform.precursor_charge for psm in psm_list])
        precursor_mzs = psm_list["precursor_mz"]
        scores = psm_list["score"]
        peptide_lengths = np.array([len(psm.peptidoform.sequence) for psm in psm_list])

        has_charge = None not in charge_states
        has_mz = None not in precursor_mzs and has_charge
        has_score = None not in scores

        if has_charge:
            charge_n = charge_states
            charge_one_hot, one_hot_names = _one_hot_encode_charge(charge_states)
            self._feature_names.extend(["charge_n"] + one_hot_names)

        if has_mz:  # Charge also required for theoretical m/z
            theo_mz = np.array([psm.peptidoform.theoretical_mz for psm in psm_list])
            abs_ms1_error_ppm = np.abs((precursor_mzs - theo_mz) / theo_mz * 10**6)
            self._feature_names.append("abs_ms1_error_ppm")

        if has_score:
            self._feature_names.append("search_engine_score")

        if has_mz and has_charge:
            experimental_mass = (precursor_mzs * charge_n) - (charge_n * 1.007276466812)
            theoretical_mass = (theo_mz * charge_n) - (charge_n * 1.007276466812)
            mass_error = experimental_mass - theoretical_mass
            self._feature_names.extend(["theoretical_mass", "experimental_mass", "mass_error"])

        self._feature_names.append("pep_len")

        for i, psm in enumerate(psm_list):
            psm.rescoring_features.update(
                dict(
                    **{"charge_n": charge_n[i]} if has_charge else {},
                    **charge_one_hot[i] if has_charge else {},
                    **{"abs_ms1_error_ppm": abs_ms1_error_ppm[i]} if has_mz else {},
                    **{"search_engine_score": scores[i]} if has_score else {},
                    **{"theoretical_mass": theoretical_mass[i]} if has_mz and has_charge else {},
                    **{"experimental_mass": experimental_mass[i]} if has_mz and has_charge else {},
                    **{"mass_error": mass_error[i]} if has_mz and has_charge else {},
                    **{"pep_len": peptide_lengths[i]},
                )
            )


def _one_hot_encode_charge(
    charge_states: np.ndarray,
) -> Tuple[Iterable[Dict[str, int]], List[str]]:
    """One-hot encode charge states."""
    n_entries = len(charge_states)
    min_charge = np.min(charge_states)
    max_charge = np.max(charge_states)

    mask = np.zeros((n_entries, max_charge - min_charge + 1), dtype=bool)
    mask[np.arange(n_entries), charge_states - min_charge] = 1
    one_hot = mask.view("i1")

    heading = [f"charge_{i}" for i in range(min_charge, max_charge + 1)]

    return [dict(zip(heading, row)) for row in one_hot], heading
