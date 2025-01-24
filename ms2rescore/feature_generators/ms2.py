"""
MS2-based feature generator.

"""

import logging
import re
from collections import defaultdict
from itertools import chain
from typing import List, Optional

import numpy as np
from psm_utils import PSMList
from rustyms import FragmentationModel, LinearPeptide, MassMode, RawSpectrum
from ms2rescore_rs import get_ms2_spectra

from ms2rescore.exceptions import ParseSpectrumError
from ms2rescore.feature_generators.base import FeatureGeneratorBase
from ms2rescore.utils import infer_spectrum_path

logger = logging.getLogger(__name__)

FRAGMENTATION_MODELS = {
    "cidhcd": FragmentationModel.CidHcd,
    "etd": FragmentationModel.Etd,
    "ethcd": FragmentationModel.Ethcd,
    "all": FragmentationModel.All,
}
MASS_MODES = {
    "average": MassMode.Average,
    "monoisotopic": MassMode.Monoisotopic,
}


class MS2FeatureGenerator(FeatureGeneratorBase):
    """DeepLC retention time-based feature generator."""

    def __init__(
        self,
        *args,
        spectrum_path: Optional[str] = None,
        spectrum_id_pattern: str = "(.*)",
        fragmentation_model: str = "All",
        mass_mode: str = "Monoisotopic",
        processes: int = 1,
        calculate_hyperscore: bool = True,  # Allow optional ?
        **kwargs,
    ) -> None:
        """
        Generate MS2-based features for rescoring.

        Parameters
        ----------

        Attributes
        ----------
        feature_names: list[str]
            Names of the features that will be added to the PSMs.

        """
        super().__init__(*args, **kwargs)

        self.spectrum_path = spectrum_path
        self.spectrum_id_pattern = spectrum_id_pattern
        self.fragmentation_model = FRAGMENTATION_MODELS[fragmentation_model.lower()]
        self.mass_mode = MASS_MODES[mass_mode.lower()]
        self.processes = processes
        self.calculate_hyperscore = calculate_hyperscore

    @property
    def feature_names(self) -> List[str]:
        return [
            "ln_explained_intensity",
            "ln_total_intensity",
            "ln_explained_intensity_ratio",
            "ln_explained_b_ion_ratio",
            "ln_explained_y_ion_ratio",
            "longest_b_ion_sequence",
            "longest_y_ion_sequence",
            "matched_b_ions",
            "matched_b_ions_pct",
            "matched_y_ions",
            "matched_y_ions_pct",
            "matched_ions_pct",
            "hyperscore",
        ]

    def add_features(self, psm_list: PSMList) -> None:
        """Add MS2-derived features to PSMs."""

        logger.info("Adding MS2-derived features to PSMs.")
        psm_dict = psm_list.get_psm_dict()
        current_run = 1
        total_runs = sum(len(runs) for runs in psm_dict.values())

        for runs in psm_dict.values():
            for run, psms in runs.items():
                logger.info(
                    f"Running MS2 for PSMs from run ({current_run}/{total_runs}) `{run}`..."
                )
                psm_list_run = PSMList(psm_list=list(chain.from_iterable(psms.values())))
                spectrum_filename = infer_spectrum_path(self.spectrum_path, run)

                self._calculate_features(psm_list_run, spectrum_filename)
                current_run += 1

    def _calculate_features(self, psm_list: PSMList, spectrum_file: str) -> List:
        """Retrieve annotated spectra for all psms."""

        spectra = get_ms2_spectra(str(spectrum_file))
        spectrum_id_pattern = re.compile(
            self.spectrum_id_pattern if self.spectrum_id_pattern else r"(.*)"
        )
        try:
            spectra_dict = {
                spectrum_id_pattern.search(spectrum.identifier).group(1): spectrum
                for spectrum in spectra
            }
        except AttributeError:
            raise ParseSpectrumError(
                "Could not parse spectrum IDs using ´spectrum_id_pattern´. Please make sure that there is a capturing in the pattern."
            )

        for psm in psm_list:
            annotated_spectrum = self._annotate_spectrum(psm, spectra_dict[psm.spectrum_id])
            psm.rescoring_features.update(
                self._calculate_spectrum_features(psm, annotated_spectrum)
            )

            if self.calculate_hyperscore:
                # Filters out peaks which are unnannotated (can be specified, but keep at b-y ions of any charge ?)
                b, y = get_by_fragments(annotated_spectrum)
                hs = calculate_hyperscore(
                    n_y=len(y), n_b=len(b), y_ion_intensities=y, b_ion_intensities=b
                )
                psm.rescoring_features.update({"hyperscore": hs})

    @staticmethod
    def _longest_ion_sequence(lst):
        max_sequence = 0
        current_sequence = 0

        for value in lst:
            current_sequence = current_sequence + 1 if value else 0
            max_sequence = max(max_sequence, current_sequence)

        return max_sequence

    def _calculate_spectrum_features(self, psm, annotated_spectrum):

        if not annotated_spectrum:
            return {}

        features = defaultdict(list)
        b_ions_matched = [False] * (len(psm.peptidoform.sequence))
        y_ions_matched = [False] * (len(psm.peptidoform.sequence))

        pseudo_count = 0.00001
        ion_fragment_regex = re.compile(r"\d+")

        for peak in annotated_spectrum:
            features["total_intensity"].append(peak.intensity)

            if peak.annotation:
                features["matched_intensity"].append(peak.intensity)
                for matched_ion in peak.annotation:
                    if "y" in matched_ion.ion:
                        features["y_ion_matched"].append(peak.intensity)
                        y_ions_matched[int(ion_fragment_regex.search(matched_ion.ion).group())] = (
                            True
                        )
                    elif "b" in matched_ion.ion:
                        features["b_ion_matched"].append(peak.intensity)
                        b_ions_matched[int(ion_fragment_regex.search(matched_ion.ion).group())] = (
                            True
                        )

        total_intensity_sum = np.sum(features["total_intensity"])
        matched_intensity_sum = np.sum(features["matched_intensity"])
        b_ion_matched_sum = np.sum(features["b_ion_matched"])
        y_ion_matched_sum = np.sum(features["y_ion_matched"])

        return {
            "ln_explained_intensity": np.log(matched_intensity_sum + pseudo_count),
            "ln_total_intensity": np.log(total_intensity_sum + pseudo_count),
            "ln_explained_intensity_ratio": np.log(
                matched_intensity_sum / total_intensity_sum + pseudo_count
            ),
            "ln_explained_b_ion_ratio": np.log(
                b_ion_matched_sum / matched_intensity_sum + pseudo_count
            ),
            "ln_explained_y_ion_ratio": np.log(
                y_ion_matched_sum / matched_intensity_sum + pseudo_count
            ),
            "longest_b_ion_sequence": self._longest_ion_sequence(b_ions_matched),
            "longest_y_ion_sequence": self._longest_ion_sequence(y_ions_matched),
            "matched_b_ions": sum(b_ions_matched),
            "matched_b_ions_pct": sum(b_ions_matched) / len(b_ions_matched),
            "matched_y_ions": sum(y_ions_matched),
            "matched_y_ions_pct": sum(y_ions_matched) / len(y_ions_matched),
            "matched_ions_pct": (sum(b_ions_matched) + sum(y_ions_matched))
            / (len(b_ions_matched) + len(y_ions_matched)),
        }

    def _annotate_spectrum(self, psm, spectrum):

        spectrum = RawSpectrum(
            title=psm.spectrum_id,
            num_scans=1,
            rt=psm.retention_time,
            precursor_charge=psm.get_precursor_charge(),
            precursor_mass=spectrum.precursor.mz,
            mz_array=spectrum.mz,
            intensity_array=spectrum.intensity,
        )
        try:
            linear_peptide = LinearPeptide(psm.peptidoform.proforma.split("/")[0])
            annotated_spectrum = spectrum.annotate(
                peptide=linear_peptide,
                model=self.fragmentation_model,
                mode=self.mass_mode,
            )
        except:  # noqa E722
            return []

        return annotated_spectrum.spectrum


def factorial(n):
    """
    Compute factorial of n using a loop.
    Parameters:
        n (int): Non-negative integer.
    Returns:
        int: Factorial of n.
    """
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result


def calculate_hyperscore(n_y, n_b, y_ion_intensities, b_ion_intensities):
    """
    Calculate the hyperscore for a peptide-spectrum match.
    Parameters:
        n_y (int): Number of matched y-ions.
        n_b (int): Number of matched b-ions.
        y_ion_intensities (list): Intensities of matched y-ions.
        b_ion_intensities (list): Intensities of matched b-ions.
    Returns:
        float: Calculated hyperscore.
    """
    # Calculate the product of y-ion and b-ion intensities
    product_y = np.sum(y_ion_intensities) if y_ion_intensities else 1
    product_b = np.sum(b_ion_intensities) if b_ion_intensities else 1

    # Calculate factorial using custom function
    factorial_y = factorial(n_y)
    factorial_b = factorial(n_b)

    # Compute hyperscore
    hyperscore = np.log(factorial_y * factorial_b * (product_y + product_b))
    return hyperscore


def infer_fragment_identity(frag, allow_ion_types=["b", "y"]):
    ion = frag.ion

    is_allowed = False
    for allowed_ion_type in allow_ion_types:
        if allowed_ion_type in ion:
            is_allowed = True
            break

    if not is_allowed:
        return False
    # Radicals
    if "·" in ion:
        return False
    if frag.neutral_loss is not None:
        return False
    if frag.charge > 2:
        return False

    return ion[0]


def get_by_fragments(annotated_spectrum):
    b_intensities = []
    y_intensities = []
    for peak in annotated_spectrum:

        for fragment in peak.annotation:

            ion_type = infer_fragment_identity(fragment)

            if ion_type == "b":
                b_intensities.append(peak.intensity)
            if ion_type == "y":
                y_intensities.append(peak.intensity)
    return b_intensities, y_intensities
