# featureExtractorTemplateMatching.py
"""Definition for the parent class for CCA, MEC, and MSI"""
from .featureExtractor import FeatureExtractor
import jax.numpy as jnp
from jax import device_put

class FeatureExtractorTemplateMatching(FeatureExtractor):
    """A parent class for CCA, MEC, and MSI"""

    __targets_frequencies_setup_guide = (
        "The frequencies of targets must be a one dimensional array of real "
        + "positive numbers, where the first element represents the "
        + "frequency of the first target, the second element is the "
        + "frequency of the second target, and so on. All frequencies must "
        + "be in Hz. ")

    def __init__(self):
        """Setting all attributes to valid initial values"""
        super().__init__()

        # Hold the pre-computed template signals for SSVE.
        self.template_signal = None
        self.harmonics_count = 0
        self.targets_frequencies = None
        self.targets_count = 0

    def compute_templates(self):
        """Pre-compute the template signals for all target frequencies"""
        t = jnp.arange(1, self.samples_count + 1)

        t = t / self.sampling_frequency

        template_signal = jnp.array(
            [[(jnp.sin(2 * jnp.pi * t * f * h), jnp.cos(2 * jnp.pi * t * f * h))
             for h in range(1, self.harmonics_count + 1)]
             for f in self.targets_frequencies])

        self.template_signal = jnp.reshape(
            template_signal,
            (self.targets_count,
             self.harmonics_count * 2,
             self.samples_count))

        self.template_signal = jnp.transpose(
            self.template_signal, axes=(0, 2, 1))

    @property
    def template_signal(self):
        """Getter function for the template signals"""
        return self._template_signal

    @template_signal.setter
    def template_signal(self, template_signal):
        """Setter function for the template signals"""
        error_message = "template_signal must be a 3D array of floats."

        if template_signal is None:
            self._template_signal = 0
            return

        try:
            template_signal = jnp.asarray(template_signal, dtype=jnp.float32)
        except (ValueError, TypeError, AttributeError):
            self.quit(error_message)

        if template_signal.ndim != 3:
            self.quit(error_message)

        # Transfer data to the desired device
        self._template_signal = device_put(template_signal, device=self.device)

    @property
    def harmonics_count(self):
        """Getter function for the number of harmonics"""
        if self._harmonics_count == 0:
            self.quit("The number of harmonics is not set properly. "
                      + "To set the number of harmonics use the "
                      + "harmonics_count option of method "
                      + "setup_feature_extractor. ")

        return self._harmonics_count

    @harmonics_count.setter
    def harmonics_count(self, harmonics_count):
        """Setter method for the number of harmonics"""
        error_message = "Number of harmonics must be a positive integer."

        try:
            harmonics_count = int(harmonics_count)
        except (ValueError, TypeError):
            self.quit(error_message)

        if harmonics_count < 0:
            self.quit(error_message)

        self._harmonics_count = harmonics_count

    @property
    def targets_frequencies(self):
        """Getter function for the frequencies of stimuli"""
        if self._targets_frequencies is None:
            self.quit("The frequencies of targets is not specified. To set "
                      + "this variable, use the targets_frequencies option "
                      + "of setup_feature_extractor. "
                      + self.__targets_frequencies_setup_guide)

        return self._targets_frequencies

    @targets_frequencies.setter
    def targets_frequencies(self, stimulation_frequencies):
        """Setter function for the frequencies of stimuli"""
        error_message = ("Target frequencies must be an array of positive "
                         + "real numbers. ")
        error_message += self.__targets_frequencies_setup_guide

        if stimulation_frequencies is None:
            self._targets_frequencies = None
            self.targets_count = 0
            return

        try:
            stimulation_frequencies = jnp.array(stimulation_frequencies)
            stimulation_frequencies = stimulation_frequencies.astype(jnp.float32)
        except (ValueError, TypeError, AttributeError):
            self.quit(error_message)

        if stimulation_frequencies.ndim == 0:
            stimulation_frequencies = jnp.array([stimulation_frequencies])

        if jnp.any(stimulation_frequencies <= 0):
            self.quit(error_message)

        self._targets_frequencies = stimulation_frequencies
        self.targets_count = stimulation_frequencies.size

    @property
    def targets_count(self):
        """Getter function for the number of targets"""
        if self._targets_count == 0:
            self.quit("The number of targets is not set. This happens "
                      + "because the target frequencies is not specified. "
                      + "To specify the target frequencies use the "
                      + "targets_frequencies option of the method "
                      + "setup_feature_extractor. ")
        return self._targets_count

    @targets_count.setter
    def targets_count(self, targets_count):
        """Setter function for the number of targets"""
        error_message = "Number of targets must be a positive integer."

        try:
            targets_count = int(targets_count)
        except (ValueError, TypeError):
            self.quit(error_message)

        if targets_count < 0:
            self.quit(error_message)

        self._targets_count = targets_count