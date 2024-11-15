# featureExtractorCCA.py
"""
Feature extraction method using correlation coefficient analysis.
Chen, Xiaogang, et al. "Filter bank canonical correlation analysis
for implementing a high-speed SSVEP-based brain–computer interface.
Journal of neural engineering 12.4 (2015): 046008.
"""

# Import the definition of the parent class.
from .featureExtractorTemplateMatching import FeatureExtractorTemplateMatching

import jax.numpy as jnp
from jax import jit, device_put, device_get

class FeatureExtractorCCA(FeatureExtractorTemplateMatching):
    """Implementation of feature extraction using CCA"""

    def __init__(self):
        "Class constructor"
        super().__init__()

        # If set to True, the feature extraction method only returns
        # the maximum correlation coefficient.  Otherwise, the class
        # returns k correlation coefficients, where k is the minimum of
        # the template signal rank and signal rank.  Most studies use
        # only the maximum.  Thus, the default value is True.
        self.max_correlation_only = True

    def setup_feature_extractor(
            self,
            harmonics_count,
            targets_frequencies,
            sampling_frequency,
            subbands=None,
            max_correlation_only=True,
            embedding_dimension=0,
            delay_step=0,
            filter_order=0,
            filter_cutoff_low=0,
            filter_cutoff_high=0,
            voters_count=1,
            random_seed=0,
            use_gpu=False,
            max_batch_size=16,
            explicit_multithreading=0,
            samples_count=0):
        """
        Setup the feature extractor parameters CCA.

        [Same docstring as before...]
        """
        self.build_feature_extractor(
            harmonics_count,
            targets_frequencies,
            sampling_frequency,
            subbands=subbands,
            embedding_dimension=embedding_dimension,
            delay_step=delay_step,
            filter_order=filter_order,
            filter_cutoff_low=filter_cutoff_low,
            filter_cutoff_high=filter_cutoff_high,
            voters_count=voters_count,
            random_seed=random_seed,
            use_gpu=use_gpu,
            max_batch_size=max_batch_size,
            explicit_multithreading=explicit_multithreading,
            samples_count=samples_count)

        self.max_correlation_only = max_correlation_only

    def get_features(self):
        """Extract features using CCA"""
        # Get the current batch of data
        signal = self.get_current_data_batch()
        correlations = self.canonical_correlation_reduced(signal)

        if self.max_correlation_only == True:
            correlations = jnp.max(correlations, axis=-1)

        batch_size = self.channel_selection_info_bundle[1]
        signals_count = correlations.shape[0] // batch_size

        # De-bundle the results.
        correlations = jnp.reshape(correlations, (
            signals_count,
            batch_size,
            self.targets_count,
            -1))

        if self.max_correlation_only == True:
            features = correlations

        else:
            # De-bundle the results.
            features = jnp.zeros((
                signals_count,
                batch_size,
                self.targets_count,
                self.features_count),
                dtype=jnp.float32)

            features = features.at[:, :, :, :correlations.shape[-1]].set(correlations)

        return features

    def get_features_multithreaded(self, signal):
        """Extract features from a single signal"""
        # Make sure signal is 3D
        signal = signal - jnp.mean(signal, axis=-1)[:, None]
        signal = signal[None, :, :]

        if self.max_correlation_only == False:
            self.features_count = jnp.min((
                self.electrodes_count, 2 * self.harmonics_count))

        correlations = self.canonical_correlation_reduced(signal)

        if self.max_correlation_only == True:
            correlations = jnp.max(correlations, axis=-1)

        # De-bundle the results.
        correlations = jnp.reshape(correlations, (
            1,
            1,
            1,
            self.targets_count,
            -1))

        if self.max_correlation_only == True:
            features = correlations

        else:
            # De-bundle the results.
            features = jnp.zeros((
                1,
                1,
                1,
                self.targets_count,
                self.features_count),
                dtype=jnp.float32)

            features = features.at[:, :, :, :, :correlations.shape[-1]].set(correlations)

        return features

    def canonical_correlation_reduced(self, signal):
        """Compute the canonical correlation between X and Y."""
        q_template = self.q_template_handle

        signal = jnp.transpose(signal, axes=(0, 2, 1))

        q_signal = jnp.linalg.qr(signal)[0]
        q_signal = jnp.transpose(q_signal, axes=(0, 2, 1))

        product = jnp.matmul(
            q_signal[:, None, :, :], q_template[None, :, :, :])

        r = jnp.linalg.svd(product, full_matrices=False, compute_uv=False)
        r = jnp.clip(r, a_min=0, a_max=1)

        return r

    def qr_decomposition(self, X):
        """QR Decomposition based on Schwarz Rutishauser algorithm"""
        Q = X
        ns, m, n = X.shape
        R = jnp.zeros((ns, n, n), dtype=jnp.float32)

        for k in jnp.arange(n):
            for i in jnp.arange(k):
                Qt = Q[:, :, i]
                R = R.at[:, i, k].set(jnp.sum(Qt * Q[:, :, k], axis=1))
                product = R[:, i, k][:, None] * Q[:, :, i]
                Q = Q.at[:, :, k].set(Q[:, :, k] - product)

            R = R.at[:, k, k].set(jnp.sqrt(jnp.sum(Q[:, :, k] ** 2, axis=-1)))
            Q = Q.at[:, :, k].set(Q[:, :, k] / R[:, k, k][:, None])

        return -Q

    def perform_voting_initialization(self):
        """Perform initialization and precomputations common to all voters"""
        # Center data
        self.all_signals = self.all_signals - jnp.mean(self.all_signals, axis=-1)[:, :, None]
        self.all_signals_handle = self.handle_generator(self.all_signals)
        rank = jnp.linalg.matrix_rank(self.all_signals)

        if jnp.any(rank < jnp.min(self.all_signals.shape[1:])):
            self.quit("Input signal is not full rank!")

        if self.max_correlation_only == False:
            self.features_count = jnp.min((
                self.electrodes_count, 2 * self.harmonics_count))

    def class_specific_initializations(self):
        """Perform necessary initializations"""
        # Perform some precomputations only in the first run.
        self.compute_templates()

        if self.samples_count == 1:
            self.quit("Signal is too short. Cannot compute canonical "
                      + "correlations of a matrix with a single sample.")

        # Center the template signal.
        self.template_signal = self.template_signal - jnp.mean(
            self.template_signal, axis=1)[:, None, :]

        # Q part of the QR decomposition
        self.q_template = jnp.zeros(
            self.template_signal.shape, dtype=jnp.float32)

        for i in jnp.arange(self.targets_count):
            self.q_template = self.q_template.at[i].set(jnp.linalg.qr(self.template_signal[i])[0])

        rank_template = jnp.linalg.matrix_rank(self.template_signal)

        if jnp.any(rank_template != 2 * self.harmonics_count):
            self.quit("Template matrix is not full rank.")

        self.template_signal_handle = self.handle_generator(
            self.template_signal)

        self.q_template_handle = self.handle_generator(
            self.q_template)

    def get_current_data_batch(self):
        """Bundle all data so they can be processed together"""
        # Extract bundle information.
        batch_index = self.channel_selection_info_bundle[0]
        batch_population = self.channel_selection_info_bundle[1]
        batch_electrodes_count = self.channel_selection_info_bundle[2]
        first_signal = self.channel_selection_info_bundle[3]
        last_signal = self.channel_selection_info_bundle[4]
        signals_count = last_signal - first_signal

        # Pre-allocate memory for the batch
        signal = jnp.zeros(
            (signals_count, batch_population,
             batch_electrodes_count, self.samples_count),
            dtype=jnp.float32)

        selected_signals = self.all_signals_handle[first_signal:last_signal]

        for j in jnp.arange(batch_population):
            current_selection = self.channel_selections[batch_index]
            signal = signal.at[:, j].set(selected_signals[:, current_selection, :])
            batch_index += 1

        signal = jnp.reshape(signal, (-1,) + signal.shape[2:])

        return signal

    @property
    def max_correlation_only(self):
        """Getter for max_correlation_only flag"""
        return self.__max_correlation_only

    @max_correlation_only.setter
    def max_correlation_only(self, flag):
        """Setter for max_correlation_only flag"""
        try:
            flag = bool(flag)
        except (ValueError, TypeError):
            self.quit("max_correlation_only flag must be Boolean.")

        self.__max_correlation_only = flag
