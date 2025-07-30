import torch
from . import PyBindMWIR

class Signal:
    def __init__(self, frequency_range = (1 * torch.pi * 1e9, 1 * torch.pi * 1e9), n_samples = 1, impl = None):
        if impl is None:
            frequency_range, n_samples = self.__SetFrequencyRange(frequency_range, n_samples)
            self.signal = PyBindMWIR.Signal(frequency_range, n_samples)
        elif type(impl) is PyBindMWIR.Signal:
            self.signal = impl
        else:
            raise TypeError("impl must be of type Signal or None")

    def SetFrequencyRange(self, frequency_range, n_samples):
        if self.signal is None:
            raise ValueError("internal signal instance has been moved.")
        frequency_range, n_samples = self.__SetFrequencyRange(frequency_range, n_samples)
        self.signal.SetFrequencyRange(frequency_range, n_samples)

    def GetFrequencyRange(self):
        if self.signal is None:
            raise ValueError("internal signal instance has been moved.")
        return self.signal.GetFrequencyRange()
    
    def GetNSamples(self):
        if self.signal is None:
            raise ValueError("internal signal instance has been moved.")
        return self.signal.GetNSamples()
    
    def GetFStep(self):
        if self.signal is None:
            raise ValueError("internal signal instance has been moved.")
        return self.signal.GetFStep()


    def __SetFrequencyRange(self, frequency_range, n_samples):
        if(type(frequency_range) is torch.Tensor):
            if(len(frequency_range.shape) != 1 or frequency_range.shape[0] != 2):
                raise ValueError("position must be a tensor of shape [2]")
        elif (type(frequency_range) is list or type(frequency_range) is tuple):
            if(len(frequency_range) != 2):
                raise ValueError("position must be a list or tuple of length 3")
        else:
            raise TypeError("position must be a torch.Tensor, list, or tuple")
        __frequency_range = torch.tensor(frequency_range, dtype=torch.float32)

        if(type(n_samples) is not int or n_samples <= 0):
            raise ValueError("n_samples must be a positive integer")
        __n_samples = torch.tensor((n_samples,), dtype=torch.int32)

        return __frequency_range, __n_samples


