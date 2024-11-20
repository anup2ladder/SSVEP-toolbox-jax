# featureExtractorMEC.py
"""
Implementation of MEC feature extraction method
Feature extraction method using minimum energy combination based on:
Friman, Ola, Ivan Volosyak, and Axel Graser. "Multiple channel
detection of steady-state visual evoked potentials for brain-computer
interfaces." IEEE transactions on biomedical engineering 54.4 (2007).
"""

from .featureExtractorTemplateMatching import FeatureExtractorTemplateMatching

import jax.numpy as jnp
from jax import jit, device_put, devices
from functools import partial
import numpy as np  # For certain functions not yet supported in JAX

class FeatureExtractorMEC(FeatureExtractorTemplateMatching):
    """Class of minimum energy combination feature extractor"""

    def __init__(self):
        """MEC feature extractor class constructor"""
        super().__init__()

        # The order of the AR model used for estimating noise energy.
        # This must be a single positive integer.
        self.ar_order = 15

        # The ratio of noise energy remained in the projected signal.
        # This must be a number between 0 and 1.
        self.energy_ratio = 0.05

        # A temporary pre-computed value.
        self.xplus = 0

        # The pseudo-inverse of each [sine, cosine] pair for each harmonic
        self.sub_template_inverse = 0

        # Device attribute to specify computation device
        self.device = None

    def setup_feature_extractor(
            self,
            harmonics_count,
            targets_frequencies,
            sampling_frequency,
            ar_order=15,
            energy_ratio=0.05,
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
        Setup the feature extractor parameters (MEC).
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

        self.ar_order = ar_order
        self.energy_ratio = energy_ratio

        # Set the computation device based on use_gpu flag
        self.set_device()

    def set_device(self):
        """Set the computation device based on use_gpu flag."""
        if self.use_gpu:
            gpu_devices = devices("gpu")
            if not gpu_devices:
                self.quit("No GPU device found, but use_gpu is set to True.")
            self.device = gpu_devices[0]
        else:
            self.device = devices("cpu")[0]

    def get_features(self):
        """Extract MEC features (SNRs) from signal"""
        # Extract current batch of data
        (signal, y_bar_squared) = self.get_current_data_batch()

        # Transfer data to the desired device
        signal = device_put(signal, device=self.device)
        y_bar_squared = device_put(y_bar_squared, device=self.device)

        # Swap the dimensions for samples and electrodes
        signal = jnp.transpose(signal, axes=(0, 2, 1))

        # Extract SNRs
        features = self.compute_snr(signal, y_bar_squared)

        batch_size = self.channel_selection_info_bundle[1]

        # De-bundle the results.
        features = jnp.reshape(features, (
            features.shape[0] // batch_size,
            batch_size,
            self.targets_count,
            self.features_count)
        )

        return features

    def get_features_multithreaded(self, signal):
        """Extract MEC features (SNRs) from signal"""
        # Signal is an E by T 2D array
        signal = signal - jnp.mean(signal, axis=-1)[:, None]
        signal = signal / jnp.std(signal, axis=-1)[:, None]
        signal = jnp.transpose(signal)
        signal = signal[None, :, :]

        # Compute Ybar per Eq. (9)
        y_bar = signal - jnp.matmul(self.xplus, signal)

        y_bar_squared = jnp.matmul(
            jnp.transpose(y_bar, axes=(0, 2, 1)), y_bar)

        y_bar_squared = y_bar_squared[None, :, :, :]

        # Transfer data to the desired device
        signal = device_put(signal, device=self.device)
        y_bar_squared = device_put(y_bar_squared, device=self.device)

        features = self.compute_snr(signal, y_bar_squared)

        # De-bundle the results.
        features = jnp.reshape(features, (
            1,
            1,
            self.targets_count,
            self.features_count))
        return features

    @partial(jit, static_argnums=(0,))
    def compute_snr(self, signal, y_bar_squared):
        """Compute the SNR"""
        # Project the signal to minimize the power of nuisance signal
        projected_signal, n_s = self.project_signal(signal, y_bar_squared)

        # Compute the signal power
        template = self.template_signal_handle[None, :, :, :]
        template = jnp.transpose(template, axes=(0, 1, 3, 2))
        power = jnp.matmul(template, projected_signal)
        power = jnp.square(power)

        # Sum sine and cosine pairs
        power = power + jnp.roll(power, -1, axis=2)
        power = power[:, :, 0:-1:2]
        power = jnp.transpose(power, axes=(2, 0, 1, 3))

        x_inverse_signal = jnp.matmul(
            self.sub_template_inverse_handle[:, None, :, :, :],
            projected_signal[None, :, :, :, :])

        x = jnp.reshape(
            self.template_signal_handle,
            self.template_signal_handle.shape[0:-1] + (-1, 2))

        x = jnp.transpose(x, (2, 0, 1, 3))
        s_bar = jnp.matmul(x[:, None, :, :, :], x_inverse_signal)
        s_bar = projected_signal[None, :, :, :, :] - s_bar
        s_bar = jnp.transpose(s_bar, axes=(0, 1, 2, 4, 3))

        # Extract the noise energy
        coefficients, noise_energy = self.yule_walker(s_bar)
        sigma_bar = self.k2 * noise_energy
        denominator = jnp.zeros(coefficients.shape[0:-1], dtype=jnp.complex64)
        coefficients = jnp.transpose(coefficients, axes=(3, 0, 1, 2, 4))
        coefficients = coefficients * (-1)

        coefficients = coefficients * self.k3_handle[None, :, None, :, :]

        denominator = jnp.sum(coefficients, axis=-1)
        denominator = jnp.transpose(denominator, axes=(1, 2, 3, 0))
        denominator = jnp.abs(1 + denominator)
        sigma_bar = sigma_bar / denominator
        power = power / sigma_bar

        # For each signal, only keep the first n_s number of channels.
        snrs = jnp.sum(power, axis=0)
        snrs = jnp.cumsum(snrs, axis=-1)

        snrs_reshaped = jnp.reshape(snrs, (-1, snrs.shape[2]))
        ns = 1 + jnp.arange(snrs_reshaped.shape[-1])
        ns = jnp.multiply(jnp.ones(snrs_reshaped.shape), ns[None, :])
        ns = (ns == (n_s.flatten())[:, None])
        snrs_reshaped = snrs_reshaped[ns]
        snrs = jnp.reshape(snrs_reshaped, snrs.shape[0:2])

        return snrs

    def project_signal(self, signal, y_bar_squared):
        """Project the signal such that noise has the minimum energy"""
        # Compute eigenvalues and eigenvectors
        eigen_values, eigen_vectors = jnp.linalg.eigh(y_bar_squared)

        # Compute how many channels we need to keep based on the desired
        # energy of the retained noise.
        n_s = self.compute_channels_count(eigen_values)

        # Normalize eigenvectors
        eigen_values = jnp.sqrt(eigen_values)
        eigen_values = jnp.expand_dims(eigen_values, axis=3)
        eigen_vectors = jnp.transpose(eigen_vectors, axes=(0, 1, 3, 2))
        eigen_vectors = eigen_vectors / eigen_values
        eigen_vectors = jnp.transpose(eigen_vectors, axes=(0, 1, 3, 2))

        # Keep maximum required number of channels
        max_index = jnp.max(jnp.array(n_s))
        eigen_vectors = eigen_vectors[:, :, :, :max_index]

        # Compute the projected signal per Eq. (7).
        projected_signal = jnp.matmul(signal[:, None, :, :], eigen_vectors)
        return projected_signal, n_s

    def compute_channels_count(self, eigen_values):
        """Compute how many channels we need based on ratio of energy."""
        running_sum = jnp.cumsum(eigen_values, axis=-1)
        total_energy = jnp.expand_dims(running_sum[:, :, -1], axis=-1)
        energy_ratio = running_sum / total_energy
        flags = (energy_ratio <= self.energy_ratio_handle)
        n_s = jnp.sum(flags, axis=-1)
        n_s = jnp.where(n_s == 0, 1, n_s)
        return n_s

    def yule_walker(self, time_series):
        """Yule-Walker AR model estimation"""
        # A short hand for ar_order
        p = self.ar_order

        r = jnp.zeros((p + 1,) + time_series.shape[0:-1], dtype=jnp.float32)
        r = r.at[0].set(jnp.sum(time_series ** 2, axis=-1))

        # Compute autocorrelation coefficients
        for k in range(1, p + 1):
            r = r.at[k].set(jnp.sum(
                time_series[..., :-k] * time_series[..., k:], axis=-1))

        r = jnp.transpose(r, axes=(1, 2, 3, 4, 0))
        r = r / self.samples_count_handle
        G = jnp.zeros(r.shape[:-1] + (p, p), dtype=jnp.float32)
        G = G.at[..., 0, :].set(r[..., :-1])

        for i in range(1, p):
            G = G.at[..., i, i:].set(r[..., :-i - 1])

        # Solve the Yule-Walker equations
        R = G + jnp.transpose(G, axes=(0, 1, 2, 3, 5, 4))
        R = R - jnp.eye(R.shape[-1])[None, None, None, None, :, :]
        rho = jnp.linalg.solve(R, r[..., 1:])
        sigmasq = jnp.sum(r[..., 1:] * rho, axis=-1)
        sigmasq = r[..., 0] - sigmasq

        return rho, sigmasq

    def perform_voting_initialization(self):
        """Perform initialization and precomputations common to all voters"""
        # Normalize all data
        self.all_signals = self.all_signals - jnp.mean(self.all_signals, axis=-1)[:, :, None]
        self.all_signals = self.all_signals / jnp.std(self.all_signals, axis=-1)[:, :, None]

        # Transfer data to the desired device
        self.all_signals_handle = device_put(self.all_signals, device=self.device)

        signal = self.all_signals_handle

        # Pre-compute y_bar per Eq. (9).
        signal = jnp.transpose(signal, axes=(0, 2, 1))

        y_bar = jnp.matmul(
            self.xplus_handle[None, :, :, :],
            signal[:, None, :, :])

        y_bar = signal[:, None, :, :] - y_bar

        self.y_bar_squared = jnp.matmul(
            jnp.transpose(y_bar, axes=(0, 1, 3, 2)), y_bar)

        # Transfer precomputed data to device
        self.y_bar_squared_handles = device_put(self.y_bar_squared, device=self.device)

    def class_specific_initializations(self):
        """Perform necessary initializations and precomputations"""
        # Perform some precomputations only in the first run.
        self.compute_templates()

        # Get the inverse of sine and cosine pairs of each harmonic.
        self.precompute_each_harmonic_inverse()

        self.xplus = jnp.matmul(
            self.template_signal,
            jnp.linalg.pinv(self.template_signal))

        # Compute constants for SNR computation
        k1 = ((-2 * 1j * jnp.pi / self.sampling_frequency)
              * self.targets_frequencies)
        k1 = k1[:, None] * jnp.arange(1, self.ar_order + 1)
        k2 = jnp.pi * self.samples_count / 4
        harmonics_scaler = jnp.arange(1, self.harmonics_count + 1)
        k3 = harmonics_scaler[:, None, None] * k1
        k3 = jnp.exp(k3)
        self.k3 = k3
        self.k2 = k2

        # Create handles and transfer to device
        self.template_signal_handle = device_put(
            self.template_signal, device=self.device)

        self.sub_template_inverse_handle = device_put(
            self.sub_template_inverse, device=self.device)

        self.xplus_handle = device_put(self.xplus, device=self.device)
        self.k2_handle = device_put(self.k2, device=self.device)
        self.k3_handle = device_put(self.k3, device=self.device)
        self.energy_ratio_handle = device_put(self.energy_ratio, device=self.device)
        self.samples_count_handle = device_put(self.samples_count, device=self.device)

    def precompute_each_harmonic_inverse(self):
        """Pre-compute the inverse of each harmonic."""
        self.sub_template_inverse = jnp.zeros(
            (self.harmonics_count,
             self.template_signal.shape[0],
             2,
             self.template_signal.shape[1]))

        for h in range(0, self.harmonics_count * 2, 2):
            x = self.template_signal[:, :, (h, h + 1)]
            self.sub_template_inverse = self.sub_template_inverse.at[h // 2].set(
                jnp.linalg.pinv(x))

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
        signal = np.zeros(
            (signals_count, batch_population,
             batch_electrodes_count, self.samples_count),
            dtype=np.float32)

        y_bar_squared = np.zeros(
            (signals_count, batch_population, self.targets_count,
             batch_electrodes_count, batch_electrodes_count),
            dtype=np.float32)

        selected_signals = self.all_signals_handle[first_signal:last_signal]
        selected_ybar = self.y_bar_squared_handles[first_signal:last_signal]

        for j in range(batch_population):
            current_selection = self.channel_selections[batch_index]
            signal[:, j] = selected_signals[:, current_selection, :]
            ybar2 = selected_ybar[:, :, current_selection, :]
            ybar2 = ybar2[:, :, :, current_selection]
            y_bar_squared[:, j] = ybar2
            batch_index += 1

        signal = np.reshape(signal, (-1,) + signal.shape[2:])
        y_bar_squared = np.reshape(
            y_bar_squared, (-1,) + y_bar_squared.shape[2:])

        # Convert to jax arrays and transfer to device
        signal = device_put(jnp.asarray(signal), device=self.device)
        y_bar_squared = device_put(jnp.asarray(y_bar_squared), device=self.device)

        return (signal, y_bar_squared)

    @property
    def ar_order(self):
        """Getter function for the order of the autoregressive model"""
        return self.__ar_order

    @ar_order.setter
    def ar_order(self, order):
        """Setter function for the order of the autoregressive model"""
        error_message = "Order of the AR model must be a positive integer."

        try:
            order = int(order)
        except (ValueError, TypeError):
            self.quit(error_message)

        if order <= 0:
            self.quit(error_message)

        self.__ar_order = order

    @property
    def energy_ratio(self):
        """Getter function for energy ratio"""
        return self.__energy_ratio

    @energy_ratio.setter
    def energy_ratio(self, energy_ratio):
        """Setter function for energy ratio"""
        error_message = "Energy ratio must be a real number between 0 and 1"

        try:
            energy_ratio = float(energy_ratio)
        except (ValueError, TypeError):
            self.quit(error_message)

        if not 0 < energy_ratio < 1:
            self.quit(error_message)

        self.__energy_ratio = energy_ratio