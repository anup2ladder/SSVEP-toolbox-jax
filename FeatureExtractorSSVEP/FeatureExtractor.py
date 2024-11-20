# featureExtractor.py
"""Definition of the class FeatureExtractor"""

import jax.numpy as jnp
import sys
from scipy.signal import sosfiltfilt, butter
from multiprocessing import Pool
import numpy as np  # For certain functions not yet supported in JAX
from jax import device_put, devices, device_get

class FeatureExtractor:
    """A parent class for all feature extraction methods"""

    # The message given to the user if all_signals is set to an invalid value.
    __all_signals_setup_guide = ("Input signals must be a 3D array with the"
        + "dimensions of (signals_count, electrodes_count, samples_count), "
        + "where signals_count represents the number of all signals to be "
        + "analyzed. If there is only one signal, then signals_count "
        + "must be set to 1 but the input signal must remain 3D. "
        + "electrodes_count is the number of electrodes, and samples_count "
        + "is the number of samples. Thus, the first dimension of the "
        + "input signal indexes the signals, the second dimension indexes "
        + "the electrodes, and the third dimension indexes the samples. ")

    __parameters_count_setup_guide = ("is not setup correctly. You are "
        + "getting this error because the variable is either None or zero. "
        + "Accessing this parameter prior to its initialization causes an "
        + "error. This parameter is inferred from all_signals. To remedy "
        + "this problem, set up the input signals (all_signals) first. ")

    __embedding_dimension_setup_guide = ("Delay embedding dimension is set "
        + "to zero but the delay_step is set to a non-zero value. Because "
        + "embedding_dimension variable is zero, the value of delay_step is "
        + "discarded. To avoid inadvertent problems, the class issues a  "
        + "warning and terminates the execution. If you want to use delay "
        + "embedding, set the embedding_dimension to a positive integer. If "
        + "you do not want to use delay embedding, set delay_step to zero. ")

    __delay_step_setup_guide = ("delay_step is set to zero while delay "
        + "embedding dimension is non-zero. A zero delay step makes delayed "
        + "signals to be similar to the non-delayed signal. Thus, including "
        + "them is pointless. If you want to use delay embedding, set "
        + "delay_step to a positive integer. If you do not want to use "
        + "delay embedding, set the embedding_dimension to zero. ")

    def __init__(self):
        """Setting all object attributes to valid initial values"""
        self.all_signals = None
        self.signals_count = 0
        self.electrodes_count = 0
        self.features_count = 1
        self.embedding_dimension = 0
        self.delay_step = 0
        self.samples_count = 0
        self.filter_order = 0
        self.cutoff_frequency_low = 0
        self.cutoff_frequency_high = 0
        self.sampling_frequency = 0
        self.subbands = None
        self.subbands_count = 1
        self.is_filterbank = False
        self.sos_matrices = None
        self.voters_count = 1
        self.random_seed = 0
        self.channel_selections = None
        self.channel_selection_info_bundle = 0
        self.use_gpu = False
        self.max_batch_size = 16
        self.explicit_multithreading = 0
        self.class_initialization_is_complete = False
        self.device = None  # Device attribute to specify computation device

    def build_feature_extractor(
            self,
            harmonics_count,
            targets_frequencies,
            sampling_frequency,
            subbands=None,
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
        """Set up the parameters of the class"""
        self.harmonics_count = harmonics_count
        self.targets_frequencies = targets_frequencies
        self.sampling_frequency = sampling_frequency
        self.subbands = subbands
        self.embedding_dimension = embedding_dimension
        self.delay_step = delay_step
        self.filter_order = filter_order
        self.cutoff_frequency_low = filter_cutoff_low
        self.cutoff_frequency_high = filter_cutoff_high
        self.random_seed = random_seed
        self.voters_count = voters_count
        self.use_gpu = use_gpu
        self.max_batch_size = max_batch_size
        self.explicit_multithreading = explicit_multithreading

        # Set the computation device based on use_gpu flag
        self.set_device()

        # Embedding delays truncates the signal.
        # Thus, samples count must be updated accordingly.
        if samples_count != 0:
            samples_count -= self.embedding_dimension * self.delay_step

        self.samples_count = samples_count
        self.construct_filters()

        # If samples_count is provided,
        if samples_count > 0:
            self.class_specific_initializations()
            self.class_initialization_is_complete = True

    def set_device(self):
        """Set the computation device based on use_gpu flag."""
        if self.use_gpu:
            gpu_devices = devices("gpu")
            if not gpu_devices:
                self.quit("No GPU device found, but use_gpu is set to True.")
            self.device = gpu_devices[0]
        else:
            self.device = devices("cpu")[0]

    def extract_features(self, all_signals):
        """
        Extract the features from all given signals.

        parameter
        ----------
        all_signals: This must a 3D numpy array with the size
        [signals_count, electrodes_count, samples_count], where
        signals_count is the number of all signals in the dataset that need
        to be processed.  If there is only one signal (e.g., online analysis),
        then the first dimension 'must' be set to 1.  electrodes_count is the
        number of channels and samples_count is the number of samples.

        returns
        ------
        all_features: This is a 5D numpy array.
        [signals, subbands, voters, targets, features].
        Starting from left, the first dimension indexes signals,
        the second dimension indexes subbands, the third dimension indexes
        voters (i.e., channel selections), the fourth dimension indexes
        targets, and the last dimension indexes features.
        For example, if the input signal (all_signals) has dimensions
        [3, 15, 1000] and the classifier is set up to use a filter bank
        with 8 subbands, 64 different random channel selections, and 40
        targets, assuming that the class generates one feature per signal,
        all_features will have a shape of [3, 8, 64, 40, 1].
        If there is only one signal to be processed, then the first dimension
        will be 1.  If the feature extractor is non-filter bank, then the
        second dimension will have a size of 1.  If no random channel
        selection is set up (i.e., the number of voters is set to 1),
        the class uses all electrodes and the third dimension will have a
        size of 1.  If there is only one feature for each signal, then
        the fifth dimension will have a size of 1.
        """

        # This calls the setter and ensures that the input
        # is properly shaped, i.e., it is a 3D numpy array.
        # This also sets signals_count, electrodes_count,
        # and samples_count automatically by extracting
        # the dimension sizes of all_signals.
        self.all_signals = all_signals

        # Filter the signal (if the user has not specified a non-zero
        # filter order, this does nothing.)
        self.bandpass_filter()

        # Create new signal by decomposing them according to the given
        # subbands. Do nothing if is_filterbank flag is False.
        self.decompose_signal()

        # Expand all_signals by adding delayed replica of it.
        # If the embedding_dimension is zero, this function does nothing.
        self.embed_time_delay()

        # Randomly pick n channel selection if the classifier is set up to use
        # voting, where n is the same as self.voters_count.  Otherwise, create
        # a single channel selection by using all electrodes.
        self.select_channels()

        # Some objects need to perform some pre-computations.
        # However, the type of pre-computations depends on the class.
        # Perform these computation only if not done so so far
        if self.class_initialization_is_complete == False:
            self.class_specific_initializations()

        # Process all signals
        features = self.process_signals_platform_agnostic()

        return features

    def get_features(self):
        """
        Extract features from signal based on the method used.
        This is an abstract method and should never be called.
        """
        pass

    def get_features_multithreaded(self, signal):
        """
        Extract features from signal based on the method used.
        This is an abstract method and should never be called.
        """
        pass

    def perform_voting_initialization(self):
        """
        Some voting operations need to be done only once.  This is a class
        dependent implementation.
        """
        pass

    def class_specific_initializations(self):
        """Perform necessary initializations"""
        pass

    def process_signals_platform_agnostic(self):
        """Process signals"""
        # Perform pre-computations that are common for all voters.
        self.perform_voting_initialization()

        features = jnp.zeros(
            (self.signals_count,
             self.voters_count,
             self.targets_count,
             self.features_count),
            dtype=jnp.float32)

        batch_index = 0

        # Get the number of electrodes in each channel selection
        selection_size = jnp.sum(self.channel_selections, axis=1)

        while batch_index < self.voters_count:
            current_electrodes_count = selection_size[batch_index]

            # How many selections have current_electrodes_count electrodes
            current_size = jnp.sum(selection_size == current_electrodes_count)

            # If less than max_batch_size, select all channel selections
            # Otherwise, pick the first max_batch_size of them.
            current_size = jnp.min(jnp.array((current_size, self.max_batch_size)))

            # Save the batch information.
            self.channel_selection_info_bundle = [
                batch_index, current_size, current_electrodes_count, 0, 0]

            # Burn the picked selections so that they don't get selected again.
            selection_size = selection_size.at[batch_index:batch_index + current_size].set(-1)

            signal_index = 0

            while signal_index < self.signals_count:
                last_signal_index = signal_index + self.max_batch_size

                last_signal_index = jnp.min(jnp.array(
                    (last_signal_index, self.signals_count)
                ))

                self.channel_selection_info_bundle[3] = signal_index
                self.channel_selection_info_bundle[4] = last_signal_index

                # Extract features
                features = features.at[
                    signal_index:last_signal_index,
                    batch_index:batch_index + current_size].set(
                    self.get_features())

                signal_index = last_signal_index

            batch_index += current_size

        features = jnp.reshape(features, (
            self.subbands_count,
            self.signals_count // self.subbands_count,
            self.voters_count,
            self.targets_count,
            self.features_count))

        features = jnp.swapaxes(features, 0, 1)

        return features

    def handle_generator(self, to_copy):
        """Copy the input and return handle"""
        # Transfer data to the desired device
        handle = device_put(to_copy, device=self.device)
        return handle

    def generate_random_selection(self):
        """Generate all random channel selections"""
        random_generator = np.random.default_rng(self.random_seed)

        random_channels_indexes = random_generator.choice(
            [True, False],
            size=(self.voters_count, self.electrodes_count),
            replace=True)

        # Ensure no empty selection
        while True:
            rows_with_zeros_only = (
                jnp.sum(random_channels_indexes, axis=1) == 0)

            if not rows_with_zeros_only.any():
                break

            random_channels_indexes[rows_with_zeros_only] = (
                random_generator.choice(
                    [True, False],
                    size=(np.sum(rows_with_zeros_only), self.electrodes_count),
                    replace=True)
                )

        return random_channels_indexes

    def select_channels(self):
        """
        If the class is set to use voting, then perform as many random
        channel selections as the number of voters and save all randomly
        selected channels.  Otherwise, use all channels.
        """
        if self.voters_count > 1:
            self.channel_selections = self.generate_random_selection()
        else:
            self.channel_selections = jnp.array([True] * self.electrodes_count)
            self.channel_selections = self.channel_selections[None, :]

        self.channel_selections = self.channel_selections.astype(bool)

        # Sort channel selections based on the number of channels
        selection_size = jnp.sum(self.channel_selections, axis=1)
        sorted_index = jnp.argsort(selection_size)
        self.channel_selections = self.channel_selections[sorted_index]

    def embed_time_delay(self):
        """Expand signal by adding delayed replicas."""
        # If no delay-embedding is requested by the user, do nothing.
        if self.embedding_dimension == 0:
            return

        # expanded_signal is the temporary growing signal
        expanded_signal = self.all_signals

        # For each embedding_dimension, add a delay replicate.
        for i in range(1, self.embedding_dimension + 1):
            start_index = i * self.delay_step

            # The signal of zeros
            tail = jnp.zeros(
                (self.signals_count, self.electrodes_count, i * self.delay_step),
                )

            # Shift the signal.
            shifted_signal = jnp.concatenate(
                [self.all_signals[:, :, start_index:], tail],
                axis=-1)

            # Stack values vertically.
            expanded_signal = jnp.concatenate([expanded_signal, shifted_signal], axis=1)

        # Truncate the signal
        expanded_signal = expanded_signal[
            :, :, :-self.delay_step * self.embedding_dimension]

        # Replace the input signal with the new delay embedded one.
        self.all_signals = expanded_signal

    def bandpass_filter(self):
        """Filter the given signal using Butterworth IIR filter"""
        if self.filter_order == 0 or self.cutoff_frequency_high == 0:
            return

        sos = butter(self.filter_order,
                     [self.cutoff_frequency_low, self.cutoff_frequency_high],
                     btype='bandpass',
                     output='sos',
                     fs=self.sampling_frequency)

        # Operate along the very last dimension of the array.
        # Since scipy functions operate on numpy arrays, we need to convert
        self.all_signals = device_get(self.all_signals)
        self.all_signals = sosfiltfilt(sos, self.all_signals, axis=-1)
        # Transfer back to device
        self.all_signals = device_put(self.all_signals, device=self.device)

    def quit(self, message="Error"):
        """A function to end the program in case of an error"""
        print("Error: " + message)
        sys.exit()

    def construct_filters(self):
        """Construct bandpass filters to be used in filterbank"""
        # If this is not filterbank, no filters needed.
        if self.is_filterbank == False:
            return

        if self.filter_order == 0:
            message = ("Filter order is zero. If you want to use "
                    + "filterbank, you must set both the filter order and "
                    + "subbands' cutoff frequencies.  If you do not want "
                    + "to  use FBCCA, do not pass subbands. ")
            self.quit(message)

        # For each given subband, create a filter with corresponding cutoff
        # frequencies.
        all_sos = []

        for band in self.subbands:
            sos = butter(self.filter_order,
                         band,
                         btype='bandpass',
                         output='sos',
                         fs=self.sampling_frequency)
            all_sos.append(sos)

        self.sos_matrices = np.array(all_sos)

    def decompose_signal(self):
        """Decompose the signal into multiple bands"""
        # If this is not filterbank, no decomposition is needed.
        if self.is_filterbank == False:
            return

        all_subbands = []

        # Since scipy functions operate on numpy arrays, we need to convert
        self.all_signals = device_get(self.all_signals)

        for filter_sos in self.sos_matrices:
            signal = sosfiltfilt(filter_sos, self.all_signals, axis=-1)
            all_subbands.append(signal)

        all_subbands = np.array(all_subbands)

        # Make sure data remains 3D.
        all_subbands = np.reshape(
            all_subbands, (-1, all_subbands.shape[2], all_subbands.shape[3]))

        # Transfer back to device
        self.all_signals = device_put(jnp.asarray(all_subbands), device=self.device)

    @property
    def all_signals(self):
        """Getter function for all signals"""
        if self.__all_signals is None:
            self.quit("all_signals is not properly set. "
                      + self.__all_signals_setup_guide)

        return self.__all_signals

    @all_signals.setter
    def all_signals(self, all_signals):
        """Setter function for all signals"""
        if all_signals is None:
            self.signals_count = 0
            self.electrodes_count = 0
            self.samples_count = 0
            self.__all_signals = None
            return

        try:
            all_signals = jnp.asarray(all_signals, dtype=jnp.float32)
        except (ValueError, TypeError, AttributeError):
            self.quit(self.__all_signals_setup_guide)

        if all_signals.ndim != 3:
            self.quit(self.__all_signals_setup_guide)

        self.__all_signals = device_put(all_signals, device=self.device)
        [self.signals_count, self.electrodes_count, self.samples_count] =\
            all_signals.shape

    # ... (Other property getters and setters remain largely the same)