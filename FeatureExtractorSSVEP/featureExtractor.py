# featureExtractor.py
"""Definition of the class FeatureExtractor"""
import numpy as np
import sys
from scipy.signal import sosfiltfilt, butter
from multiprocessing import Pool
import platform

# Try importing CuPy for NVIDIA GPU support
try:
    import cupy as cp
    cupy_available_global = True
except ImportError:
    cupy_available_global = False
    cp = np

# Try importing MLX for Apple Silicon GPU support
try:
    import mlx.core as mx
    mlx_available_global = platform.processor() == 'arm' and platform.system() == 'Darwin'
except ImportError:
    mlx_available_global = False
    mx = np

class BackendManager:
    """Manages array computation backend selection and operations"""
    
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        self._setup_backend()
        
    def _setup_backend(self):
        """Initialize the appropriate backend based on hardware and availability"""
        if not self.use_gpu:
            self.backend = 'numpy'
            self.xp = np
            return
            
        # Check for Apple Silicon first
        if mlx_available_global:
            self.backend = 'mlx'
            self.xp = mx
        # Then check for NVIDIA GPU
        elif cupy_available_global:
            self.backend = 'cupy'
            self.xp = cp
        else:
            self.backend = 'numpy'
            self.xp = np
            
    def to_numpy(self, array):
        """Convert array to numpy if needed"""
        if self.backend == 'cupy':
            return cp.asnumpy(array)
        elif self.backend == 'mlx':
            return array.numpy()
        return array
    
    def to_device(self, array):
        """Convert numpy array to device array"""
        if self.backend == 'cupy':
            return cp.asarray(array)
        elif self.backend == 'mlx':
            return mx.array(array)
        return array
    

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
        # All the signals that need to be processed.  This must always be a 3D
        # numpy array with dimensions equal to the [signals_count, 
        # electrodes_count, samples_count].  If there is only one signal to be
        # processed, the array must still be 3D with the size of the first 
        # dimension being 1.  If a 2D array is passed, the class issues an
		# error and terminates execution.
        self.all_signals = None 
        
        # Total number of signals.  This variable is extracted automatically 
        # from the first dimension of all_signals.  This must be always a 
        # natural number.  If there is only one signal (e.g., during an online 
        # analysis), this variable is set to 1.  During batch processing (e.g., 
        # offline analysis of a dataset), this variable is automatically set
        # to the number of signals to be processed. 
        self.signals_count = 0
        
        # The number of channels or electrodes.  This must be a natural number.
        # This variable is set automatically based on the second dimension of
        # variable all_signals. 
        self.electrodes_count = 0
        
        # The number of features each feature extractor generates. 
        # Must be a positive integer. 
        self.features_count = 1
                                  
        # This is the dimension of time-delay embedding. 
        # This must be a non-negative integer.  If set to zero, no time-dely
        # embedding will be used.
        # If there are E electrodes and we set the embedding_dimension to 
        # n, the class expands the input signal as if we had n*E channels. 
        # The additional channels are generated by shift_left operators. 
        # The number of samples that we shift each signal is 
        # controlled by delay_step.  Embedding delays truncates the signal. 
        # Make sure the signal is long enough. 
        self.embedding_dimension = 0
        self.delay_step = 0
        
        # The number of samples in each signal (i.e., signal length is seconds
        # times the sampling rate).  This variable is extracted automatically
        # from the last dimension of variable all_signals.  This variable must
        # always be a natural number.                 
        self.samples_count = 0
        
        # The order of the filter to be used for filtering signals.
        # This must be a single positive ineger.
        self.filter_order = 0
        
        # Low and high cutoff frequencies for the filter (in Hz). 
        # These must be real positive numbers. 
        self.cutoff_frequency_low = 0
        self.cutoff_frequency_high = 0
        
        # The sampling rate of the signal (in samples per second). 
        # It must be a real positive value. 
        self.sampling_frequency = 0
        
        # This must be a 2D array with size S by 2, where S is the 
        # number of subbands to decompose the signal to.  The first
        # and second elements in each row describe the first and second
        # cutoff frequencies in Hz of the bandpass filter that is used to 
        # create subbands.  The number of subbands is inferred implicitly
        # from the number of rows in this matrix. 
        self.subbands = None
        self.subbands_count = 1
        self.is_filterbank = False
        
        # The sos for all matrices that we use in the filterbank. 
        # Saving them enables us to compute filters only once. 
        # Thus, improving the performance. 
        self.sos_matrices = None
        
        # The number of electrode-selections that are used for
        # classification.  This is the same as the number of voters.  If
        # votersCount is larger that the cardinality of the power set of 
        # the current selected electrodes, then at least one combination is
        # bound to happen more than once.  However, because the selection is
        # random, even if that's not the case, repettitions are still
        # possible.  If unset or set to an invalid value, no voting will 
        # be used. 
        self.voters_count = 1
        
        # The seed for random number generator that controls the electrodes
        # selection for each classification.  Use this for re-producibility.
        self.random_seed = 0
        
        # The random channels (electrodes) that are used for extracting 
        # features.  If used with voting, this represent the channel-selection
        # of each voter.  If the class is set up to not use voting, this 
        # is ignored.  
        self.channel_selections = None
                
        # A variable to include other information about the current 
        # chanel selection, e.g., the number of selections in each batch,
        # the index of the first selection, etc. It can be a tuple if a class
        # needs more information. 
        self.channel_selection_info_bundle = 0
        
        # A Boolean flag to instruct whether to use GPU or not.  
        # When set to false, no GPU is used.  When set to True, a GPU is used. 
        self.use_gpu = False
               
        # The maximum number of signals/channel selections that are 
        # processed together.  Increasing the batch size helps with 
        # parallelization but it also increases memory requirements. 
        self.max_batch_size = 16
        
        # This parameter determines whether to use explicit multithreading
        # or not.  If set to a non-positive integer, no multithreading will
        # be used. If set to a positive integer, the class creates multiple
        # threads to process signals/voters in parallel.  The number of 
        # threads is the same as the value of this variable.  E.g., if
        # set to 4, the class distributes the workload among four threads. 
        # Typically, this parameter should be the same as the number of cores
        # the cpu has, if multithreading is to be used. 
        # Multithreading cannot be used when use_gpu is set to True.
        self.explicit_multithreading = 0
        
        # A falg variable that allows us to see if class-specific 
        # initializations are completed or not.  class-specific initialization
        # can be performed when the class is being set up provided that the
        # user provides the number of samples. If the user does not provide
        # the number of samples, the class cannot perform class-specific 
        # initializations during class set up. Instead, it waits for the 
        # user to call extract_features(data). The class then automatically
        # extracts the number of samples from data and performs class specific
        # initialization done. The latter option can have a negative impact
        # on the performance if extract_features(data) is being called 
        # in a loop or periodically (e.g., in BCI2000 loop).
        self.class_initialization_is_complete = False
        
        # The number of devices       
        if cupy_available_global == True:        
            self.devices_count = cp.cuda.runtime.getDeviceCount()
        else:
            self.devices_count = 0
            
        # Initialize backend manager
        self.backend = BackendManager()
        
        # Update device detection
        if cupy_available_global:
            self.nvidia_devices_count = cp.cuda.runtime.getDeviceCount()
        else:
            self.nvidia_devices_count = 0
            
        self.has_apple_silicon = mlx_available_global
            
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
        """Set up the parameters of the calss"""
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
        
        # Process all signals using self.explicit_multithreading number of
        # threads
        if self.explicit_multithreading > 0:           
            self.use_gpu = False
            features = self.process_signals_multithreaded()
        else:        
            features = self.process_signals_platform_agnostic()       
        
        if self.use_gpu == True:
            cp.cuda.Stream.null.synchronize()         
        
        return features
   
    def get_features(self, device):
        """
        Extract features from signal based on the method used.
        This is an abstract method and should never be called. 
        """        
        # This method is abstract. It is implementation
        # depends on the actual feature extraction method.
        # All non-abstract classes, inhereting form this
        # class must implement this method. 
        pass
    
    def get_features_multithreaded(self, signal):
        """
        Extract features from signal based on the method used.
        This is an abstract method and should never be called. 
        """        
        # This method is abstract. It is implementation
        # depends on the actual feature extraction method.
        # All non-abstract classes, inhereting form this
        # class must implement this method. 
        pass
    
    def perform_voting_initialization(self, device=0):
        """
        Some voting operations need to be done only once.  This is a class
        dependent implementation.  Thu, the method needs to be made concrete 
        by each sub-class.  These intializations and precomputings can lead to 
        substantial speed ups.        
        """
        # Empty
        # Must be made concrete by the sub-class.
        pass
    
    def class_specific_initializations(self):
        """Perform necessary initializations"""
        # Abstract method        
        # Do nothing
        pass
    
    def process_signals_platform_agnostic(self):
        """Process signals on GPU or CPU depending on use_gpu flag"""
        # Perform pre-computations that are common for all voters.
        if self.use_gpu == True:
            device = 1
            xp = cp
        else:
            device = 0
            xp = np
            
        self.perform_voting_initialization(device)  

        features = xp.zeros(
            (self.signals_count, 
             self.voters_count, 
             self.targets_count,              
             self.features_count),
            dtype=xp.float32)  
                  
        batch_index = 0    
        
        # Get the number of electrodes in each channel selection
        selection_size = np.sum(self.channel_selections, axis=1)
        
        while batch_index < self.voters_count:
            current_electrodes_count = selection_size[batch_index]
            
            # How many selections have current_electrodes_count electrodes                                   
            current_size = np.sum(selection_size == current_electrodes_count)
            
            # If less than max_batch_size, select all channel selections
            # Otherwise, pick the first max_batch_size of them. 
            current_size = np.min((current_size, self.max_batch_size))
            
            # Save the batch information.  We later use these to extract 
            # the data of each batch. 
            self.channel_selection_info_bundle = [
                batch_index, current_size, current_electrodes_count, 0, 0]
            
            # Burn the picked selections so that they don't get selected again.
            selection_size[batch_index:batch_index+current_size] = -1
            
            signal_index = 0
            
            while signal_index < self.signals_count:
                last_signal_index = signal_index + self.max_batch_size
                
                last_signal_index = np.min(
                    (last_signal_index, self.signals_count)
                    )
                
                self.channel_selection_info_bundle[3] = signal_index
                self.channel_selection_info_bundle[4] = last_signal_index
                               
                # Extract features
                features[
                    signal_index:last_signal_index,
                    batch_index:batch_index+current_size] = (
                    self.get_features(device))
                                        
                signal_index = last_signal_index
                       
            batch_index += current_size
          
        if self.use_gpu == True:
            features = cp.asnumpy(features)
        
        features = np.reshape(features, (            
            self.subbands_count,
            self.signals_count//self.subbands_count,
            self.voters_count,
            self.targets_count,
            self.features_count))
        
        features = np.swapaxes(features, 0, 1)
        
        return features
    
    def process_signals_multithreaded(self):
        """Process each signal/voter in a thread"""
        tasks = np.arange(0, self.voters_count * self.signals_count)
   
        with Pool(self.explicit_multithreading) as pool:
            features = pool.map(
                self.extract_features_multithreaded, tasks)    
    
        features = np.array(features)              
        features = features[:, 0, 0, 0, :, :]
        
        # De-bundle voters from signals 
        features = np.reshape(
            features, 
            (self.voters_count, self.signals_count) + features.shape[1:]
            )
        
        features = np.transpose(features, axes=(1, 0, 2, 3))
        
        # De-bundle subbands from signals 
        original_signals_count = self.signals_count//self.subbands_count
        features = np.reshape(
            features,
            (self.subbands_count, original_signals_count) + features.shape[1:]
            )
        
        features = np.swapaxes(features, 0, 1)
               
        return features
    
    def extract_features_multithreaded(self, idx):
        """The feature extraction done by each thread"""        
        # Use thread ID to determine which signal and which electrode 
        # selection should the thread process.
        channel_index, signal_index = np.unravel_index(
            idx, (self.voters_count, self.signals_count))
                        
        # Select the signal and channels for this thread.
        signal = self.all_signals[
            signal_index,
            self.channel_selections[channel_index]]
        
        # Extract features
        features = self.get_features_multithreaded(signal)

        return features
                
    def handle_generator(self, to_copy):
        """Copy the input on every device and return handles for each device"""        
        if self.use_gpu == False:
            handle = [to_copy]
            return handle
        
        handle = []        
        to_copy = cp.asnumpy(to_copy)
        
        # The first handle is always the CPU handle
        handle.append(to_copy)
        
        if to_copy.dtype == cp.float64:
            to_copy = cp.float32(to_copy)
            
        elif to_copy.dtype == cp.complex128:
            to_copy = cp.complex64(to_copy)
                    
        # Copy the data in each device
        for i in range(self.devices_count):
            with cp.cuda.Device(i):
                handle.append(cp.asarray(to_copy))
                
        return handle
    
    def generate_random_selection(self):
        """Generate all random channel selections"""
        random_generator = np.random.default_rng(self.random_seed)
        
        random_channels_indexes = np.zeros(
            (self.voters_count, self.electrodes_count))
                
        # The while loop ensures there is no empty selection
        while True:
            
            # Flag all empty electrode selections
            rows_with_zeros_only = (
                np.sum(random_channels_indexes, axis=1) == 0)
            
            if not rows_with_zeros_only.any():
                break
                        
            # For each voter, draw logical indexes of the electrodes
            # the voter uses. 
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
             self.channel_selections = np.array([True]*self.electrodes_count)
             self.channel_selections = np.expand_dims(
                 self.channel_selections, axis=0)
             
        self.channel_selections = self.channel_selections.astype(bool)    
        
        # Sort channel selections based on the number of channels
        selection_size = np.sum(self.channel_selections, axis=1)
        sorted_index = np.argsort(selection_size)
        self.channel_selections = self.channel_selections[sorted_index]
     
    def embed_time_delay(self):
        """Expand signal by adding delayed replica of it.
        Include time-shifted copies of the signal. Each replica is 
        created by delaying all channels by delayStep samples. 
        The number of replica is determined by embeddingDimension."""                      
        # If no delay-embedding is requested by the user, do nothing.
        if self.embedding_dimension == 0:
            return
        
        # expanded_signal is the temporary growing signal, to which we add
        # delayed replica one by one. 
        expanded_signal = self.all_signals
        
        # For each embedding_dimension, add a delay replicate.
        for i in range(1, self.embedding_dimension+1):
            start_index = i * self.delay_step
            
            # The signal of zero that contains exactly as many zeros as needed
            # to keep the size of the signal the same after the shift. 
            # tail will be added at the end (right side) of the signal.
            tail = np.zeros(
                (self.signals_count, self.electrodes_count, i*self.delay_step),
                )
            
            # Shift the signal.  Append the right number of zeros for lost
            # samples.
            shifted_signal =  np.block(
                [self.all_signals[:, :, start_index:], tail]
                )
            
            # Stack values vertically.
            expanded_signal = np.block([[expanded_signal], [shifted_signal]])
        
        # Get rid of all zeros.  Effectively, we are truncating the signal.  
        expanded_signal = expanded_signal[
            :, :, :-self.delay_step*self.embedding_dimension]
        
        # Replace the input signal with the new delay embedded one.
        # Update all other paramters (e.g., electrodes count, samples count,
        # etc.)
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
        self.all_signals = sosfiltfilt(sos, self.all_signals, axis=-1)

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
            message =("Filter order is zero. If you want to use "
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

        # Run single-threaded
        if self.explicit_multithreading <= 0:
            all_subbands = []              
            
            for filter_sos in self.sos_matrices:            
                signal = sosfiltfilt(filter_sos, self.all_signals, axis=-1)
                all_subbands.append(signal)
        
        # Multi-threaded version
        else:        
            tasks = np.arange(self.subbands_count)
            
            with Pool(self.explicit_multithreading) as pool:
                all_subbands = pool.map(
                    self.decompose_signal_thread_task, tasks
                    )
        
        all_subbands = np.array(all_subbands)
        
        # Make sure data remains 3D. 
        all_subbands = np.reshape(
            all_subbands, (-1, all_subbands.shape[2], all_subbands.shape[3]))
        
        self.all_signals = all_subbands
    
    def decompose_signal_thread_task(self, task_index):
        """Decompose all signals for the subband indexed by task_index"""   
        # Multithreading over filters typically yields to better CPU 
        # utilization.
        filter_sos = self.sos_matrices[task_index]
        signal = sosfiltfilt(filter_sos, self.all_signals, axis=-1)                   
        return signal
            
    def get_array_module(self, array):
        """Return the module of the array even if failed to import cupy"""
        return self.backend.xp
        
    def filterbank_standard_aggregator(self, features, a=1.25, b=0.25, axis=1):
        """
        Aggregates the features extracted by filterbank into a single number.
        Input features can be matrix with any shape but the subbands must
        be in the axis dimension. 
        """
        subbands_count = features.shape[axis]
        
        # If there is only one subband, there is no need for aggregation
        if subbands_count == 1:
            return features
        
        # Set up weights 
        n = 1 + np.arange(0, subbands_count)
        w = n**(-a) + b
        
        features = np.moveaxis(features, axis, -1)
        shape = features.shape
        features = np.reshape(features, (-1, subbands_count))
        features = np.multiply(w[None, :], np.square(features))
        features = np.reshape(features, shape)
        features = np.sum(features, axis=-1)
        features = np.expand_dims(features, axis=axis)
        return features
    
    def voting_classification_by_count(self, features):
        """
        Give each target a score based on number of votes.
        The input matrix must be 2D. The first dimension must index the
        channel selections while the second dim must index features. 
        """
        if features.ndim != 2:
            print("Could not get the features based on votes count. "
                  + "The input features matrix must be 2D. The first "
                  + "dimension must index the channel selections "
                  + "while the second dimension must index the features. "
                  + "Returning the input features without modifying it. "
                )

        winner_targets = np.argmax(features, axis=1)
        
        targets_count = features.shape[1]        
        features_based_on_votes = np.zeros((1, targets_count))
        
        for target in np.arange(targets_count):
            features_based_on_votes[0, target] = np.sum(
                winner_targets == target)
            
        return features_based_on_votes        
    
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
            all_signals = all_signals.astype(np.float32)
        except (ValueError, TypeError, AttributeError):
            self.quit(self.__all_signals_setup_guide)
            
        if all_signals.ndim != 3:
            self.quit(self.__all_signals_setup_guide)
            
        self.__all_signals = all_signals
        [self.signals_count, self.electrodes_count, self.samples_count] =\
            all_signals.shape 
    
    @property
    def signals_count(self):
        """Getter function for the number of signals"""        
        if self._signals_count == 0:
            self.quit("signlas_count " 
                      + self.__parameters_count_setup_guide 
                      + self.__all_signals_setup_guide)
            
        return self._signals_count
    
    @signals_count.setter
    def signals_count(self, signals_count):
        """Setter function for the number of signals"""
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
        """Getter function for the number of electrodes"""
        if self._electrodes_count == 0:
            self.quit("electrodes_count " 
                      + self.__parameters_count_setup_guide 
                      + self.__all_signals_setup_guide)
            
        return self._electrodes_count

    @electrodes_count.setter
    def electrodes_count(self, electrodes_count):
        """Setter function for the number of electrodes"""
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
        """Getter function for class attribute features_count"""
        if self.__features_count <= 0:
            self.quit(
                "Trying to access features_count before initializing "
                + "it. ")
        return self.__features_count
    
    @features_count.setter
    def features_count(self, features_count):
        """Setter function for class attribute features_count"""
        error_message = ("feautres_count must be a positive integer. ")
        
        try:
            features_count = int(features_count)
        except(ValueError, TypeError):
            self.quit(error_message)
        
        if features_count <= 0:
            self.quit(error_message)
        
        self.__features_count = features_count
        
    @property
    def samples_count(self):
        """Getter function for the number of samples"""
        if self.__samples_count == 0:
            self.quit("samples_count " 
                      + self.__parameters_count_setup_guide 
                      + self.__all_signals_setup_guide)
            
        return self.__samples_count
   
    @samples_count.setter
    def samples_count(self, samples_count):
        """Setter function for the number of samples"""
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
                    + "current samples_count. This has probably happended "
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
        """Getter function for attribute embedding_dimension"""
        if self.__embedding_dimension == 0 and self.__delay_step != 0:
            self.quit(self.__embedding_dimension_setup_guide)
            
        return self.__embedding_dimension
    
    @embedding_dimension.setter
    def embedding_dimension(self, embedding_dimension):
        """Setter function for attribute embedding_dimension"""
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
        """Getter function for the attribute delay_step"""
        if self.__delay_step == 0 and self.__embedding_dimension != 0:
            self.quit(self.__delay_step_setup_guide)
            
        return self.__delay_step
    
    @delay_step.setter
    def delay_step(self, delay_step):
        """Setter function for attribute delay_step"""
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
        """Getter function for the attribute filter_order"""   
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
        """Setter function for the attribute filter_order"""
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
        """Getter function for the first cutoff frequency of the filter"""
        if self.__cutoff_frequency_low > self.__cutoff_frequency_high:
            self.quit("The first cutoff frequency cannot exceed the "
                      + "second one. ")
            
        return self.__cutoff_frequency_low
    
    @cutoff_frequency_low.setter
    def cutoff_frequency_low(self, cutoff_frequency):
        """Setter function for the first cutoff frequency of the filter"""
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
        """Getter function for the second cutoff frequency of the filter"""
        if self.__cutoff_frequency_low > self.__cutoff_frequency_high:
            self.quit("The first cutoff frequency cannot exceed the "
                      + "second one. ")

        return self.__cutoff_frequency_high
    
    @cutoff_frequency_high.setter
    def cutoff_frequency_high(self, cutoff_frequency):
        """Setter function for the second cutoff frequency of the filter"""
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
        """Getter function for sampling frequency"""
        if self.__sampling_frequency == 0:
            self.quit("Sampling frequency is not set. You can setup the "
                      + " sampling frequency using the sampling_frequency "
                      + "option of setup_feature_extractor method. ")
            
        return self.__sampling_frequency
    
    @sampling_frequency.setter
    def sampling_frequency(self, frequency):
        """Setter function for sampling frequency"""
        error_message = "Sampling frequency must a be a non-negative integer."
        
        try:
            frequency = float(frequency)
        except (TypeError, ValueError):
            self.quit(error_message)
        
        if frequency < 0:
            self.quit(error_message)           
            
        self.__sampling_frequency = frequency
        
    @property
    def sos_matrices(self):
        """Getter function for sos_matrices"""
        return self.__sos_matrices
    
    @sos_matrices.setter
    def sos_matrices(self, matrices):
        """Setter functioni for sos_matrices"""
        
        if matrices is None:
            self.__sos_matrices = 0
            return
        
        try:
            matrices = matrices.astype(float)
        except(ValueError, TypeError, AttributeError):
            self.quit("SOS matrix of the filter must be an array of floats.")
        
        self.__sos_matrices = matrices
        
    @property 
    def subbands(self):
        """Getter function for class attribute subbands"""        
        return self.__subbands
    
    @subbands.setter 
    def subbands(self, subbands):
        """Setter function for class attribute subbands"""
        message = ("Subbands must be a matrix of real nonnegative "
                + "numbers. The row corresponds to one subband. Each "
                + "row must have two column, where the first column is "
                + "the first cutoff frequency and the second column is "
                + "the second cutoff frequency. The entry in the second "
                + "must be larger than the entry in the first column "
                + "for each row.");
        
        if subbands is None:
            self.__subbands = None
            self.is_filterbank = False
            self.subbands_count = 1
            return
        
        # Chek if subbands is an array
        try:
            subbands = subbands.astype(np.float32)                
        except (ValueError, TypeError, AttributeError):
            self.quit(message)
        
        # subbands must be a 2D array
        if (subbands < 0).any() or subbands.ndim != 2:
            self.quit(message)
        
        # Check if all second cutoff frequencies are larger than first ones
        if (subbands[:, 0] >= subbands[:, 1]).any():
            self.quit("Second cutoff of the BPF must exceed the first one")
         
        # It is up to the user to make sure that cutoff frequencies are
        # consistent and make sense. 
        if np.sum(subbands) == 0:
            self.is_filterbank = False
        else:
            self.is_filterbank = True
        
        self.subbands_count = (subbands.shape)[0]
        self.__subbands = subbands
        
    
    @property
    def subbands_count(self):
        """Getter function for class attribute subbands_count"""
        return self.__subbands_count
    
    @subbands_count.setter 
    def subbands_count(self, subbands_count):
        """Setter function for the attribute subbands_count"""
        message = "The number of subbands must be a positive integer.  "
        
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
        """Getter function for class attribute voters_count"""
        return self.__voters_count
    
    @voters_count.setter
    def voters_count(self, voters_count):
        """Setter function for the attribute voters_count"""
        message = "The number of voters must be a positive integer.  "
        
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
        """Getter function for the attribute random_seed"""
        return self.__random_seed
    
    @random_seed.setter
    def random_seed(self, random_seed):
        """Setter function for the attribute random_seed"""        
        message = "random seed must be a non negative integer.  "
        
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
        """Getter function for the attribute channel_selections"""
        return self.__channel_selections 
    
    @channel_selections.setter
    def channel_selections(self, channel_selections):
        """Setter function for the attribute channel_selections"""        
        message = ("channel selections is not set properly. Do not set "
        + "up this variable directly.  ")
        
        if channel_selections is None:
            self.__channel_selections = None
            return
            
        try:
            channel_selections = np.bool8(channel_selections)
        except(TypeError, ValueError):
            self.quit(message)

        self.__channel_selections = channel_selections
        
    @property
    def use_gpu(self):
        """Getter function for the attribute use_gpu"""
        return self.__use_gpu
    
    @use_gpu.setter
    def use_gpu(self, flag):
        """Setter function for the attribute use_gpu"""
        message = "Cannot set use_gpu. use_gpu flag must either True or False."
        
        try:
            flag = np.bool8(flag)
        except(TypeError, ValueError):
            self.quit(message)
            
        if flag.size != 1:
            self.quit(message)
            
        if flag == True and self.explicit_multithreading > 0:
            self.quit(
                "Cannot set use_gpu because explicit_multithreading is set "
                + "to a positive value.  use_gpu is not available when "
                + "multithreading is enabled. ")
            
        if flag == True and cupy_available_global == False:
            self.quit(
                "Cannot set use_gpu because the calss failed to import cupy. "
                + "This is probably because cupy is not installed correctly. "
                + "Or the host does not have any CUDA-capable device. "
                + "You can still run this code even if the host does not "
                + "a CUDA device or even if cupy is not installed. "
                + "But in order to do this, you should set use_gpu flag "
                + "in setup_feature_extractor() function to false. ")                
            
        self.__use_gpu = flag
        self.backend = BackendManager(flag)
        
    @property
    def max_batch_size(self):
        """Getter function for the attribute max_batch_size"""
        return self.__max_batch_size
    
    @max_batch_size.setter
    def max_batch_size(self, max_batch_size):
        """Setter function for the attribute max_batch_size"""
        message = "max_batch_size must be a positive integer.  "
        
        try:
            max_batch_size = np.int32(max_batch_size)
        except(ValueError, TypeError):
            self.quit(message)
            
        if max_batch_size.size != 1:
            self.quit(message)
            
        if max_batch_size <= 0:
            self.quit(message)
            
        self.__max_batch_size = max_batch_size
        
    @property
    def explicit_multithreading(self):
        """Getter function for the attribute explicit_multithreading"""
        return self.__explicit_multithreading
    
    @explicit_multithreading.setter
    def explicit_multithreading(self, cores_count):
        """Setter function for the attribute explicit_multithreading"""
        message = "explicit_multithreading must be an integer. "
        
        try:
            cores_count = np.int32(cores_count)
        except(ValueError, TypeError):
            self.quit(message)
            
        if cores_count.size != 1:
            self.quit(message)
            
        if cores_count < 0:
            cores_count = 0
            
        if cores_count >= 2048:
            self.quit(
                "explicit_multithreading is too large.  Typically " 
                + "this should be the same size as the number of cores " 
                + "or a number in that order. ")
        
        if self.use_gpu == True and cores_count > 0:
            self.quit(
                "Cannot set explicit_multithreading when use_gpu "
                + "is set to True.  Multithreading is not supported "
                + "when using GPUs. ")
            
        self.__explicit_multithreading = cores_count