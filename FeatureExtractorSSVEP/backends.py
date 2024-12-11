import numpy as np
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = np

try:
    import mlx.core as mx
    import platform
    MLX_AVAILABLE = platform.processor() == 'arm' and platform.system() == 'Darwin'
except ImportError:
    MLX_AVAILABLE = False
    mx = np

class BackendBase:
    """Base class for array computation backends"""
    def eval(self, x):
        return x

class NumPyBackend(BackendBase):
    """NumPy backend implementation"""
    def __init__(self):
        self.xp = np
        
    def asarray(self, x):
        return np.asarray(x)
        
    def to_numpy(self, x):
        return x

class CuPyBackend(BackendBase):
    """CuPy backend implementation"""
    def __init__(self):
        self.xp = cp
        
    def asarray(self, x):
        return cp.asarray(x)
        
    def to_numpy(self, x):
        return cp.asnumpy(x)

class MLXBackend(BackendBase):
    """MLX backend with lazy evaluation handling"""
    def __init__(self):
        self.xp = mx
        
    def asarray(self, x):
        return mx.array(x)
        
    def to_numpy(self, x):
        return x.numpy()
        
    def eval(self, x):
        """Force evaluation of lazy computations"""
        if hasattr(x, 'eval'):
            return x.eval()
        return x

def get_backend(use_gpu=False):
    """Get appropriate backend based on hardware and settings"""
    if not use_gpu:
        return NumPyBackend()
    
    if MLX_AVAILABLE:
        return MLXBackend()
    elif CUPY_AVAILABLE:
        return CuPyBackend()
    
    return NumPyBackend()
