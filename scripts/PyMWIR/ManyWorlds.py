import torch
from . import PyBindMWIR

class ManyWorlds:
    def __init__(self, extent = torch.tensor([[-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1]]), resolution = 0.001, impl = None):
        extent = self.__SetExtent(extent)
        resolution = self.__SetResolution(resolution)
    
        # self.occupancy = torch.zeros((int((extent[0, 1] - extent[0, 0]) / resolution),
        #                               int((extent[1, 1] - extent[1, 0]) / resolution),
        #                               int((extent[2, 1] - extent[2, 0]) / resolution)),
        #                              dtype=torch.float32)

    def GenerateMesh(self):
        pass

    def __SetExtent(self, extent):
        if(type(extent) is torch.Tensor):
            if(len(extent.shape) != 2 or extent.shape[0] != 3 or extent.shape[1] != 2):
                raise ValueError("extent must be a tensor of shape [3, 2]")
        elif (type(extent) is list or type(extent) is tuple):
            if(len(extent) != 3):
                raise ValueError("position must be a list or tuple of length 3")
            if(not all(isinstance(x, (list, tuple)) and len(x) == 2 for x in extent)):
                raise ValueError("each element of extent must be a list or tuple of length 2")
        else:
            raise TypeError("position must be a torch.Tensor, list, or tuple")
        return torch.tensor(extent, dtype=torch.float32)

    def __SetResolution(self, resolution):
        if(type(resolution) is not float and type(resolution) is not int):
            raise TypeError("resolution must be a float or int")
        if(resolution <= 0):
            raise ValueError("resolution must be greater than 0")
        return float(resolution)