# featureExtractor.py
"""Definition of the class FeatureExtractor"""
import sys
from scipy.signal import sosfiltfilt, butter
from multiprocessing import Pool
import numpy as np  # Keep numpy for some array creation operations not in MLX
import mlx.core as mx  # Main MLX library


# Force CPU usage at module level
mx.set_default_device(mx.cpu)


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
        + "discarded. To avoid inadvartant problems, the classs issues a  "
        + "warning and terminates the executuion. If you want to use delay, "
        + "embedding set the embedding_dimension to a positive integer. If "
        + "you do not want to use delay embedding, set delay_step to zero. ")
    
    __delay_step_setup_guide = ("delay_step is set to zero while delay "
        + "embedding dimension is non-zero. A zero delay step makes delayed "
        + "signals to be similar to the non-delayed signal. Thus, including "
        + "them is pointless. If you want to use delay embedding, set "
        + "delay_step to a positive integer. If you do not want to use "
        + "delay embedding, set the embedding_dimension to zero. ")
        
    def __init__(self):
        """Setting all object attributes to valid initial values"""
        
        # Force CPU usage with MLX for this instance
        mx.set_default_device(mx.cpu)
        
        # All the signals that need to be processed.  This must always be a 3D
        # array with dimensions equal to the [signals_count, 
        # electrodes_count, samples_count].  If there is only one signal to be
        # processed, the array must still be 3D with the size of the first 
        # dimension being 1.  If a 2D array is passed, the class issues an
        # error and terminates execution.
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
        
        # Filterbank related attributes
        self.subbands = None
        self.subbands_count = 1
        self.is_filterbank = False
        self.sos_matrices = None
        
        # Voting related attributes
        self.voters_count = 1
        self.random_seed = 0
        self.channel_selections = None
        self.channel_selection_info_bundle = 0
        
        # Processing flags
        self.use_gpu = False  # Kept for compatibility but not used with MLX
        self.max_batch_size = 16
        self.explicit_multithreading = 0
        self.class_initialization_is_complete = False
            
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
        self.use_gpu = use_gpu  # Kept for compatibility
        self.max_batch_size = max_batch_size
        self.explicit_multithreading = explicit_multithreading
        
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
                        
    def extract_features(self, all_signals):
        """Extract features from all given signals."""
        self.all_signals = all_signals
        self.bandpass_filter()
        self.decompose_signal()
        self.embed_time_delay()
        self.select_channels()   
        
        if self.class_initialization_is_complete == False:            
            self.class_specific_initializations()    
        
        if self.explicit_multithreading > 0:           
            features = self.process_signals_multithreaded()
        else:        
            features = self.process_signals_platform_agnostic()       
        
        return features
   
    def get_features(self, device):
        """Abstract method: Extract features based on the method used."""
        pass
    
    def get_features_multithreaded(self, signal):
        """Abstract method: Extract features based on the method used."""
        pass
    
    def perform_voting_initialization(self, device=0):
        """Abstract method: Initialize voting operations."""
        pass
    
    def class_specific_initializations(self):
        """Abstract method: Perform necessary initializations"""
        pass
    
    def process_signals_platform_agnostic(self):
        """Process signals using MLX"""
        self.perform_voting_initialization()  

        # np.zeros -> mx.zeros
        features = mx.zeros(
            (self.signals_count, 
             self.voters_count, 
             self.targets_count,              
             self.features_count),
            dtype=mx.float32)  
                  
        batch_index = 0    
        
        # np.sum -> mx.sum
        selection_size = mx.sum(self.channel_selections, axis=1)
        
        while batch_index < self.voters_count:
            current_electrodes_count = selection_size[batch_index]
            
            # np.sum -> mx.sum
            current_size = mx.sum(selection_size == current_electrodes_count)
            
            # np.min -> mx.minimum
            current_size = mx.minimum(current_size, self.max_batch_size)
            
            self.channel_selection_info_bundle = [
                batch_index, current_size, current_electrodes_count, 0, 0]
            
            selection_size = mx.array([
                -1 if i >= batch_index and i < batch_index + current_size 
                else s for i, s in enumerate(selection_size)])
            
            signal_index = 0
            
            while signal_index < self.signals_count:
                last_signal_index = signal_index + self.max_batch_size
                
                last_signal_index = mx.minimum(
                    last_signal_index, self.signals_count)
                
                self.channel_selection_info_bundle[3] = signal_index
                self.channel_selection_info_bundle[4] = last_signal_index
                               
                features = features.at[
                    signal_index:last_signal_index,
                    batch_index:batch_index+current_size].set(
                    self.get_features(0))
                                        
                signal_index = last_signal_index
                       
            batch_index += current_size
          
        # Reshape operations
        features = mx.reshape(features, (            
            self.subbands_count,
            self.signals_count//self.subbands_count,
            self.voters_count,
            self.targets_count,
            self.features_count))
        
        # np.swapaxes -> mx.transpose with explicit axes
        features = mx.transpose(features, (1, 0, 2, 3, 4))
        
        return features
    
    def process_signals_multithreaded(self):
        """Process each signal/voter in a thread"""
        # np.arange -> mx.arange
        tasks = mx.arange(0, self.voters_count * self.signals_count)
   
        with Pool(self.explicit_multithreading) as pool:
            features = pool.map(
                self.extract_features_multithreaded, tasks)    
    
        features = mx.array(features)              
        features = features[:, 0, 0, 0, :, :]
        
        features = mx.reshape(
            features, 
            (self.voters_count, self.signals_count) + features.shape[1:]
            )
        
        features = mx.transpose(features, (1, 0, 2, 3))
        
        original_signals_count = self.signals_count//self.subbands_count
        features = mx.reshape(
            features,
            (self.subbands_count, original_signals_count) + features.shape[1:]
            )
        
        features = mx.transpose(features, (1, 0, 2, 3))
               
        return features

    def extract_features_multithreaded(self, idx):
        """The feature extraction done by each thread"""        
        # np.unravel_index -> manual calculation with MLX
        channel_index = idx // self.signals_count
        signal_index = idx % self.signals_count
                        
        # Select the signal and channels for this thread
        current_selection = self.channel_selections[channel_index]
        # MLX equivalent of boolean indexing
        selected_indices = mx.array([i for i, select in enumerate(current_selection) if select])
        signal = self.all_signals[signal_index][selected_indices]
        
        # Extract features
        features = self.get_features_multithreaded(signal)
        return features
                
    def handle_generator(self, to_copy):
        """Generate array handle - simplified for MLX"""
        # MLX handles device placement automatically
        # Just convert to MLX array if it isn't already
        if not isinstance(to_copy, mx.array):
            to_copy = mx.array(to_copy)
        return [to_copy]
    
    def generate_random_selection(self):
        """Generate all random channel selections"""
        # np.random.default_rng -> mx.random.seed
        mx.random.seed(self.random_seed)
        
        # np.zeros -> mx.zeros
        random_channels_indexes = mx.zeros(
            (self.voters_count, self.electrodes_count))
                
        while True:
            # Check for rows with zeros only
            # np.sum -> mx.sum
            rows_with_zeros_only = (
                mx.sum(random_channels_indexes, axis=1) == 0)
            
            if not mx.any(rows_with_zeros_only):
                break
                        
            # Generate random boolean array
            # Using MLX's random.rand and thresholding for boolean values
            zero_rows = mx.sum(rows_with_zeros_only)
            random_values = mx.random.rand(
                zero_rows, self.electrodes_count) < 0.5
            
            # Update only the rows that are all zeros
            zero_indices = mx.where(rows_with_zeros_only)[0]
            for i, idx in enumerate(zero_indices):
                random_channels_indexes = random_channels_indexes.at[idx].set(random_values[i])
        
        return random_channels_indexes
    
    def select_channels(self):
        """Select channels for processing"""
        if self.voters_count > 1:
            self.channel_selections = self.generate_random_selection()
        else:
            # Create array of True values
            self.channel_selections = mx.ones(self.electrodes_count, dtype=mx.bool_)
            # Add extra dimension
            self.channel_selections = mx.expand_dims(self.channel_selections, 0)
             
        # Cast to boolean
        self.channel_selections = mx.array(self.channel_selections, dtype=mx.bool_)
        
        # Sort channel selections based on number of channels
        selection_size = mx.sum(self.channel_selections, axis=1)
        sorted_indices = mx.argsort(selection_size)
        self.channel_selections = self.channel_selections[sorted_indices]
     
    def embed_time_delay(self):
        """Expand signal by adding delayed replica of it."""
        if self.embedding_dimension == 0:
            return
        
        # Start with original signal
        expanded_signal = self.all_signals
        
        for i in range(1, self.embedding_dimension+1):
            start_index = i * self.delay_step
            
            # Create zero padding
            # np.zeros -> mx.zeros
            tail = mx.zeros(
                (self.signals_count, self.electrodes_count, i*self.delay_step))
            
            # Shift and concatenate
            shifted_signal = mx.concatenate([
                self.all_signals[:, :, start_index:],
                tail
            ], axis=2)
            
            # Stack vertically
            expanded_signal = mx.concatenate([expanded_signal, shifted_signal], axis=0)
        
        # Truncate zeros
        expanded_signal = expanded_signal[
            :, :, :-self.delay_step*self.embedding_dimension]
        
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
        
        # Convert to numpy for scipy, then back to MLX
        np_signals = mx.array(self.all_signals).numpy()
        filtered = sosfiltfilt(sos, np_signals, axis=-1)
        self.all_signals = mx.array(filtered)

    def quit(self, message="Error"):
        """Error handling function"""
        print("Error: " + message)
        sys.exit()
        
    def construct_filters(self):
        """Construct bandpass filters for filterbank"""
        if not self.is_filterbank:
            return 
        
        if self.filter_order == 0:
            message = ("Filter order is zero. If you want to use "
                    + "filterbank, you must set both the filter order and "
                    + "subbands' cutoff frequencies.  If you do not want "
                    + "to use FBCCA, do not pass subbands. ")
            self.quit(message)
            
        # Create filters for each subband
        all_sos = []
        for band in self.subbands:            
            sos = butter(self.filter_order,
                         band,
                         btype='bandpass',
                         output='sos',
                         fs=self.sampling_frequency)            
            all_sos.append(sos)
            
        self.sos_matrices = mx.array(all_sos)
        
    def decompose_signal(self):
        """Decompose the signal into multiple bands"""
        if not self.is_filterbank:
            return

        if self.explicit_multithreading <= 0:
            all_subbands = []              
            
            for filter_sos in self.sos_matrices:
                # Convert to numpy for scipy filtering
                np_signals = mx.array(self.all_signals).numpy()
                filtered = sosfiltfilt(filter_sos.numpy(), np_signals, axis=-1)
                all_subbands.append(mx.array(filtered))
        
        else:        
            tasks = mx.arange(self.subbands_count)
            
            with Pool(self.explicit_multithreading) as pool:
                all_subbands = pool.map(
                    self.decompose_signal_thread_task, tasks)
        
        all_subbands = mx.array(all_subbands)
        
        # Reshape to maintain 3D structure
        all_subbands = mx.reshape(
            all_subbands, (-1, all_subbands.shape[2], all_subbands.shape[3]))
        
        self.all_signals = all_subbands
    
    def decompose_signal_thread_task(self, task_index):
        """Thread task for signal decomposition"""
        filter_sos = self.sos_matrices[task_index]
        # Convert to numpy for scipy filtering
        np_signals = mx.array(self.all_signals).numpy()
        filtered = sosfiltfilt(filter_sos.numpy(), np_signals, axis=-1)
        return mx.array(filtered)
        
    def filterbank_standard_aggregator(self, features, a=1.25, b=0.25, axis=1):
        """Aggregate filterbank features"""
        subbands_count = features.shape[axis]
        
        if subbands_count == 1:
            return features
        
        # Create weights
        n = 1 + mx.arange(0, subbands_count)
        w = mx.power(n, -a) + b
        
        # Move axis to end for easier computation
        features = mx.moveaxis(features, axis, -1)
        shape = features.shape
        features = mx.reshape(features, (-1, subbands_count))
        
        # Apply weights
        features = mx.multiply(w[None, :], mx.square(features))
        features = mx.reshape(features, shape)
        features = mx.sum(features, axis=-1)
        features = mx.expand_dims(features, axis=axis)
        return features
    
    def voting_classification_by_count(self, features):
        """Classification based on vote counting"""
        if features.ndim != 2:
            print("Could not get the features based on votes count. "
                  + "The input features matrix must be 2D. The first "
                  + "dimension must index the channel selections "
                  + "while the second dimension must index the features. "
                  + "Returning the input features without modifying it. ")
            return features

        winner_targets = mx.argmax(features, axis=1)
        
        targets_count = features.shape[1]        
        features_based_on_votes = mx.zeros((1, targets_count))
        
        # Count votes for each target
        for target in mx.arange(targets_count):
            features_based_on_votes = features_based_on_votes.at[0, target].set(
                mx.sum(winner_targets == target))
            
        return features_based_on_votes

    # Property definitions remain largely unchanged, just updating array operations
    @property
    def all_signals(self):
        """Getter for all signals"""
        if self.__all_signals is None:
            self.quit("all_signals is not properly set. " 
                      + self.__all_signals_setup_guide)
            
        return self.__all_signals
    
    @all_signals.setter
    def all_signals(self, all_signals):
        """Setter for all signals""" 
        if all_signals is None:
            self.signals_count = 0
            self.electrodes_count = 0
            self.samples_count = 0
            self.__all_signals = None
            return
        
        try:
            # Convert to MLX array with float32 dtype
            all_signals = mx.array(all_signals, dtype=mx.float32)
        except (ValueError, TypeError, AttributeError):
            self.quit(self.__all_signals_setup_guide)
            
        if len(all_signals.shape) != 3:
            self.quit(self.__all_signals_setup_guide)
            
        self.__all_signals = all_signals
        self.signals_count = all_signals.shape[0]
        self.electrodes_count = all_signals.shape[1]
        self.samples_count = all_signals.shape[2]

    # Other property definitions follow similar pattern...
    # The logic remains the same, just updating array operations to use MLX
    @property
    def signals_count(self):
        """Getter for number of signals"""        
        if self._signals_count == 0:
            self.quit("signals_count " 
                      + self.__parameters_count_setup_guide 
                      + self.__all_signals_setup_guide)
            
        return self._signals_count
    
    @signals_count.setter
    def signals_count(self, signals_count):
        """Setter for number of signals"""
        error_message = "signals_count must be a non-negative integer. "   
        
        try:
            signals_count = int(signals_count)            
        except (ValueError, TypeError):
            self.quit(error_message)
        
        if signals_count < 0:
            self.quit(error_message)
                    
        self._signals_count = signals_count
        
    @property
    def electrodes_count(self):
        """Getter for number of electrodes"""
        if self._electrodes_count == 0:
            self.quit("electrodes_count " 
                      + self.__parameters_count_setup_guide 
                      + self.__all_signals_setup_guide)
            
        return self._electrodes_count

    @electrodes_count.setter
    def electrodes_count(self, electrodes_count):
        """Setter for number of electrodes"""
        error_message = "electrodes_count must be a positive integer. "        
        try:
            electrodes_count = int(electrodes_count)
        except (ValueError, TypeError):
            self.quit(error_message)
        
        if electrodes_count < 0:
            self.quit(error_message)
            
        self._electrodes_count = electrodes_count
        
    @property
    def features_count(self):
        """Getter for features_count"""
        if self.__features_count <= 0:
            self.quit(
                "Trying to access features_count before initializing "
                + "it. ")
        return self.__features_count
    
    @features_count.setter
    def features_count(self, features_count):
        """Setter for features_count"""
        error_message = ("features_count must be a positive integer. ")
        
        try:
            features_count = int(features_count)
        except(ValueError, TypeError):
            self.quit(error_message)
        
        if features_count <= 0:
            self.quit(error_message)
        
        self.__features_count = features_count
        
    @property
    def samples_count(self):
        """Getter for number of samples"""
        if self.__samples_count == 0:
            self.quit("samples_count " 
                      + self.__parameters_count_setup_guide 
                      + self.__all_signals_setup_guide)
            
        return self.__samples_count
   
    @samples_count.setter
    def samples_count(self, samples_count):
        """Setter for number of samples"""
        error_message = "samples_count count must be a positive integer. "
        
        try:
            samples_count = int(samples_count)
        except (ValueError, TypeError):
            self.quit(error_message)
            
        if samples_count < 0:
            self.quit(error_message)
            
        try:            
            if (self.__samples_count != 0 
                and samples_count != self._samples_count):
                self.quit(
                    "Inconsistent samples count. It seems that the new "
                    + "samples_count is non-zero and different from the "
                    + "current samples_count. This has probably happened "
                    + "because the samples_count variable set in "
                    + "setup_feature_extractor() is different from the size "
                    + "of the third dimension of signals provided in "
                    + "extract_features function. If you do not know the "
                    + "samples_count before having the signal consider "
                    + "removing samples_count option in extract_features "
                    + "function. If you know samples_count before having the "
                    + "signal, make sure it is consistent with "
                    + "dimensionality of the signal. ")
        except(AttributeError):
            self.__samples_count = samples_count    
            return
            
        self.__samples_count = samples_count       
        
    @property
    def embedding_dimension(self):
        """Getter for embedding_dimension"""
        if self.__embedding_dimension == 0 and self.__delay_step != 0:
            self.quit(self.__embedding_dimension_setup_guide)
            
        return self.__embedding_dimension
    
    @embedding_dimension.setter
    def embedding_dimension(self, embedding_dimension):
        """Setter for embedding_dimension"""
        error_message = "Delay embedding dim. must be a non-negative integer. "
        
        try:
            embedding_dimension = int(embedding_dimension)
        except(TypeError, ValueError):
            self.quit(error_message)
        
        if embedding_dimension < 0:
            self.quit(error_message)
            
        self.__embedding_dimension = embedding_dimension
        
    @property
    def delay_step(self):
        """Getter for delay_step"""
        if self.__delay_step == 0 and self.__embedding_dimension != 0:
            self.quit(self.__delay_step_setup_guide)
            
        return self.__delay_step
    
    @delay_step.setter
    def delay_step(self, delay_step):
        """Setter for delay_step"""
        error_message = "Delay step size must be a positive integer. "
        
        try:
            delay_step = int(delay_step)
        except(ValueError, TypeError):
            self.quit(error_message)
            
        if delay_step < 0:
            self.quit(error_message)
            
        self.__delay_step = delay_step
        
    @property
    def filter_order(self):
        """Getter for filter_order"""   
        if self._filter_order == 0 and (
                self.cutoff_frequency_low != 0 or
                self.cutoff_frequency_high != 0):
            self.quit("filter_order is zero but the cutoff frequencies are "
                      + "non-zero. To use bandpass filtering, set the "
                      + "filter_order to a positive integer. To not use "
                      + "bandpass filtering, set the cutoff frequencies to "
                      + "zero. ")
            
        return self._filter_order
    
    @filter_order.setter 
    def filter_order(self, filter_order):
        """Setter for filter_order"""
        message = "The order of the filter must be a positive integer. "
        
        try:
            filter_order = int(filter_order)
        except(TypeError, ValueError):
            self.quit(message)
        
        if filter_order < 0:
            self.quit(message)                    
        
        self._filter_order = filter_order   
        
    @property
    def cutoff_frequency_low(self):
        """Getter for low cutoff frequency"""
        if self.__cutoff_frequency_low > self.__cutoff_frequency_high:
            self.quit("The first cutoff frequency cannot exceed the "
                      + "second one. ")
            
        return self.__cutoff_frequency_low
    
    @cutoff_frequency_low.setter
    def cutoff_frequency_low(self, cutoff_frequency):
        """Setter for low cutoff frequency"""
        message = "First cutoff frequency must be a positive real number. "
        
        try:
            cutoff_frequency = float(cutoff_frequency)
        except(ValueError, TypeError):
            self.quit(message)
        
        if cutoff_frequency < 0:
            self.quit(message)
        
        self.__cutoff_frequency_low = cutoff_frequency
            
    @property
    def cutoff_frequency_high(self):
        """Getter for high cutoff frequency"""
        if self.__cutoff_frequency_low > self.__cutoff_frequency_high:
            self.quit("The first cutoff frequency cannot exceed the "
                      + "second one. ")

        return self.__cutoff_frequency_high
    
    @cutoff_frequency_high.setter
    def cutoff_frequency_high(self, cutoff_frequency):
        """Setter for high cutoff frequency"""
        message = "Second cutoff frequency must be a positive real number. "
        
        try:
            cutoff_frequency = float(cutoff_frequency)
        except(ValueError, TypeError):
            self.quit(message)
        
        if cutoff_frequency < 0:
            self.quit(message)
        
        self.__cutoff_frequency_high = cutoff_frequency
        
    @property
    def sampling_frequency(self):
        """Getter for sampling frequency"""
        if self.__sampling_frequency == 0:
            self.quit("Sampling frequency is not set. You can setup the "
                      + " sampling frequency using the sampling_frequency "
                      + "option of setup_feature_extractor method. ")
            
        return self.__sampling_frequency
    
    @sampling_frequency.setter
    def sampling_frequency(self, frequency):
        """Setter for sampling frequency"""
        error_message = "Sampling frequency must be a non-negative number."
        
        try:
            frequency = float(frequency)
        except (TypeError, ValueError):
            self.quit(error_message)
        
        if frequency < 0:
            self.quit(error_message)           
            
        self.__sampling_frequency = frequency
        
    @property
    def sos_matrices(self):
        """Getter for sos_matrices"""
        return self.__sos_matrices
    
    @sos_matrices.setter
    def sos_matrices(self, matrices):
        """Setter for sos_matrices"""
        if matrices is None:
            self.__sos_matrices = 0
            return
        
        try:
            # Convert to MLX array
            matrices = mx.array(matrices, dtype=mx.float32)
        except(ValueError, TypeError, AttributeError):
            self.quit("SOS matrix of the filter must be an array of floats.")
        
        self.__sos_matrices = matrices
        
    @property 
    def subbands(self):
        """Getter for subbands"""        
        return self.__subbands
    
    @subbands.setter 
    def subbands(self, subbands):
        """Setter for subbands"""
        message = ("Subbands must be a matrix of real nonnegative "
                + "numbers. The row corresponds to one subband. Each "
                + "row must have two columns, where the first column is "
                + "the first cutoff frequency and the second column is "
                + "the second cutoff frequency. The entry in the second "
                + "must be larger than the entry in the first column "
                + "for each row.")
        
        if subbands is None:
            self.__subbands = None
            self.is_filterbank = False
            self.subbands_count = 1
            return
        
        try:
            # Convert to MLX array
            subbands = mx.array(subbands, dtype=mx.float32)
        except (ValueError, TypeError, AttributeError):
            self.quit(message)
        
        # Check dimensions and values
        if mx.any(subbands < 0) or len(subbands.shape) != 2:
            self.quit(message)
        
        if mx.any(subbands[:, 0] >= subbands[:, 1]):
            self.quit("Second cutoff of the BPF must exceed the first one")
         
        if mx.sum(subbands) == 0:
            self.is_filterbank = False
        else:
            self.is_filterbank = True
        
        self.subbands_count = subbands.shape[0]
        self.__subbands = subbands
        
    @property
    def subbands_count(self):
        """Getter for subbands_count"""
        return self.__subbands_count
    
    @subbands_count.setter 
    def subbands_count(self, subbands_count):
        """Setter for subbands_count"""
        message = "The number of subbands must be a positive integer."
        
        if subbands_count is None:
            self.__subbands_count = 1
            return 
            
        try:
            subbands_count = int(subbands_count)
        except(ValueError, TypeError):
            self.quit(message)
            
        if subbands_count <= 0:
            self.quit(message)
            
        self.__subbands_count = subbands_count
        
    @property
    def voters_count(self):
        """Getter for voters_count"""
        return self.__voters_count
    
    @voters_count.setter
    def voters_count(self, voters_count):
        """Setter for voters_count"""
        message = "The number of voters must be a positive integer."
        
        if voters_count is None:
            self.__voters_count = 1
            return
            
        try:
            voters_count = int(voters_count)
        except(ValueError, TypeError):
            self.quit(message)
        
        if voters_count <= 0:
            self.quit(message)
        
        self.__voters_count = voters_count
        
    @property
    def random_seed(self):
        """Getter for random_seed"""
        return self.__random_seed
    
    @random_seed.setter
    def random_seed(self, random_seed):
        """Setter for random_seed"""        
        message = "random seed must be a non negative integer."
        
        if random_seed is None:
            self.__random_seed = 0
            return
            
        try:
            random_seed = int(random_seed)
        except(TypeError, ValueError):
            self.quit(message)
            
        if random_seed < 0:
            self.quit(message)
            
        self.__random_seed = random_seed
        
    @property
    def channel_selections(self):
        """Getter for channel_selections"""
        return self.__channel_selections 
    
    @channel_selections.setter
    def channel_selections(self, channel_selections):
        """Setter for channel_selections"""        
        message = ("channel selections is not set properly. Do not set "
        + "up this variable directly.")
        
        if channel_selections is None:
            self.__channel_selections = None
            return
            
        try:
            # Convert to MLX boolean array
            channel_selections = mx.array(channel_selections, dtype=mx.bool_)
        except(TypeError, ValueError):
            self.quit(message)

        self.__channel_selections = channel_selections
        
    @property
    def use_gpu(self):
        """Getter for use_gpu flag"""
        return self.__use_gpu
    
    @use_gpu.setter
    def use_gpu(self, flag):
        """Setter for use_gpu flag"""
        message = "use_gpu flag must be either True or False."
        
        try:
            flag = bool(flag)
        except(TypeError, ValueError):
            self.quit(message)
            
        if self.explicit_multithreading > 0:
            self.quit(
                "Cannot set use_gpu because explicit_multithreading is set "
                + "to a positive value.")
            
        self.__use_gpu = flag
        
    @property
    def max_batch_size(self):
        """Getter for max_batch_size"""
        return self.__max_batch_size
    #Continue here
    @max_batch_size.setter
    def max_batch_size(self, max_batch_size):
        """Setter for max_batch_size"""
        message = "max_batch_size must be a positive integer."
        
        try:
            max_batch_size = int(max_batch_size)
        except(ValueError, TypeError):
            self.quit(message)
            
        if max_batch_size <= 0:
            self.quit(message)
            
        self.__max_batch_size = max_batch_size
        
    @property
    def explicit_multithreading(self):
        """Getter for explicit_multithreading"""
        return self.__explicit_multithreading
    
    @explicit_multithreading.setter
    def explicit_multithreading(self, cores_count):
        """Setter for explicit_multithreading"""
        message = "explicit_multithreading must be an integer."
        
        try:
            cores_count = int(cores_count)
        except(ValueError, TypeError):
            self.quit(message)
            
        if cores_count < 0:
            cores_count = 0
            
        if cores_count >= 2048:
            self.quit(
                "explicit_multithreading is too large. Typically "
                + "this should be the same size as the number of cores "
                + "or a number in that order.")
        
        if self.use_gpu == True and cores_count > 0:
            self.quit(
                "Cannot set explicit_multithreading when use_gpu "
                + "is set to True. Multithreading is not supported "
                + "when using GPUs.")
            
        self.__explicit_multithreading = cores_count