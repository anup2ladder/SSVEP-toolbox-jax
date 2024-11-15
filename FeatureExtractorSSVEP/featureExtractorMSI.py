# featureExtractorMSI.py
"""
Implementation of MSI feature extractor.
Zhang, Yangsong, et al. "The extension of multivariate
synchronization index method for SSVEP-based BCI." Neurocomputing
269 (2017): 226-231
"""

from .featureExtractorTemplateMatching import FeatureExtractorTemplateMatching
import jax.numpy as jnp

class FeatureExtractorMSI(FeatureExtractorTemplateMatching):
    """Class of MSI feature extractor"""

    def __init__(self):
        """MSI feature extractor class constructor"""
        super().__init__()

        # This is the covariance matrix of the template SSVEP.
        # We can pre-compute this once to improve performance.
        self.C22 = 0

    def setup_feature_extractor(
            self,
            harmonics_count,
            targets_frequencies,
            sampling_frequency,
            embedding_dimension=0,
            delay_step=0,
            filter_order=0,
            filter_cutoff_low=0,
            filter_cutoff_high=0,
            subbands=None,
            voters_count=1,
            random_seed=0,
            use_gpu=False,
            max_batch_size=16,
            explicit_multithreading=0,
            samples_count=0):
        """
        Setup the feature extractor parameters (MSI).
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

    def get_features(self):
        """Extract MSI features (synchronization indexes) from signal"""
        signal = self.get_current_data_batch()

        features = self.compute_synchronization_index(signal)
        batch_size = self.channel_selection_info_bundle[1]

        features = jnp.reshape(features, (
            features.shape[0] // batch_size,
            batch_size,
            self.targets_count,
            self.features_count)
        )
        return features

    def get_features_multithreaded(self, signal):
        """Extract MSI features from a single signal"""
        # Make sure signal is 3D
        signal = signal - jnp.mean(signal, axis=-1)[:, None]
        signal = signal[None, :, :]
        features = self.compute_synchronization_index(signal)

        # De-bundle the results.
        features = jnp.reshape(features, (
            1,
            1,
            self.targets_count,
            self.features_count)
        )

        return features

    def compute_synchronization_index(self, signal):
        """Compute the synchronization index between signal and templates"""
        electrodes_count = signal.shape[1]
        r_matrix = self.compute_r(signal)

        # Keep only the eigenvalues
        eigen_values = jnp.linalg.eigh(r_matrix)[0]
        eigen_values = eigen_values / (
            2 * self.harmonics_count_handle + electrodes_count)

        score = eigen_values * jnp.log(eigen_values)
        score = jnp.sum(score, axis=-1)
        score = score / jnp.log(r_matrix.shape[-1])
        score = score + 1
        return score

    def compute_r(self, signal):
        """Compute matrix R as explained in Eq. (7)"""
        C11 = self.get_data_covariance(signal)

        C11 = C11[:, None, :, :]
        signal = signal[:, None, :, :]

        C12 = jnp.matmul(
            signal, self.template_signal_handle[None, :, :, :])

        C12 = C12 / self.samples_count_handle
        electrodes_count = C11.shape[-1]

        # Eq. (6)
        upper_left = jnp.matmul(C11, C12)
        upper_left = jnp.matmul(upper_left, self.C22_handle)

        lower_right = jnp.matmul(
            self.C22_handle, jnp.transpose(C12, axes=[0, 1, 3, 2]))

        lower_right = jnp.matmul(lower_right, C11)
        eye1 = jnp.eye(electrodes_count, dtype=jnp.float32)

        eye1 = eye1 + jnp.zeros(
            upper_left.shape[0:2] + eye1.shape, dtype=jnp.float32)

        eye2 = jnp.eye(C12.shape[-1], dtype=jnp.float32)

        eye2 = eye2 + jnp.zeros(
            upper_left.shape[0:2] + eye2.shape, dtype=jnp.float32)

        part1 = jnp.concatenate((eye1, upper_left), axis=-1)
        part2 = jnp.concatenate((lower_right, eye2), axis=-1)
        r_matrix = jnp.concatenate((part1, part2), axis=-2)

        return r_matrix.real

    def get_data_covariance(self, signal):
        """Compute the covariance of data per Eq. (3) and Eq. (6)"""
        C11 = jnp.matmul(signal, jnp.transpose(signal, axes=(0, 2, 1)))
        C11 = C11 / self.samples_count_handle
        C11 = jnp.linalg.inv(C11)
        C11 = self.matrix_square_root(C11)
        return C11

    def matrix_square_root(self, A):
        """Compute the square root of a matrix using eigenvalue decomposition"""
        w, v = jnp.linalg.eigh(A)
        D = jnp.diag(jnp.sqrt(w))
        sqrt_A = v @ D @ jnp.linalg.inv(v)
        return sqrt_A

    def perform_voting_initialization(self):
        """Perform initialization and precomputations common to all voters"""
        # Center data
        self.all_signals = self.all_signals - jnp.mean(self.all_signals, axis=-1)[:, :, None]
        self.all_signals_handle = self.handle_generator(self.all_signals)

    def class_specific_initializations(self):
        """Perform necessary initializations"""
        # Perform some precomputations only in the first run.
        self.compute_templates()
        self.precompute_template_covariance()

        # Create handles
        self.template_signal_handle = self.handle_generator(
            self.template_signal)
        self.C22_handle = self.handle_generator(self.C22)

        self.harmonics_count_handle = self.handle_generator(
            self.harmonics_count)

        self.samples_count_handle = self.handle_generator(
            jnp.float32(self.samples_count))

    def precompute_template_covariance(self):
        """Pre-compute and save the covariance matrix of the template"""
        # Eq. (4)
        C22 = jnp.matmul(
            jnp.transpose(self.template_signal, axes=[0, 2, 1]),
            self.template_signal)

        C22 = C22 / self.samples_count

        # Eq. (6)
        C22 = jnp.linalg.inv(C22)
        self.C22 = self.matrix_square_root(C22)

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

        for j in range(batch_population):
            current_selection = self.channel_selections[batch_index]
            signal = signal.at[:, j].set(selected_signals[:, current_selection, :])
            batch_index += 1

        signal = jnp.reshape(signal, (-1,) + signal.shape[2:])

        return signal