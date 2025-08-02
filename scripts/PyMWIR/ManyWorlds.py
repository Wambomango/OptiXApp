import torch
from . import PyBindMWIR

class ManyWorlds:
    def __init__(self, extent = torch.tensor([[-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1]]), resolution = 0.001, impl = None):
        extent = self.__SetExtent(extent)
        resolution = self.__SetResolution(resolution)
        self.many_worlds = PyBindMWIR.ManyWorlds(extent, resolution)

    def SetExtent(self, extent):
        if self.many_worlds is None:
            raise ValueError("ManyWorlds ownership has been transferred")
        extent = self.__SetExtent(extent)
        self.many_worlds.SetExtent(extent)

    def SetResolution(self, resolution):
        if self.many_worlds is None:
            raise ValueError("ManyWorlds ownership has been transferred")
        resolution = self.__SetResolution(resolution)
        self.many_worlds.SetResolution(resolution)

    def GetExtent(self):
        if self.many_worlds is None:
            raise ValueError("ManyWorlds ownership has been transferred")
        return self.many_worlds.GetExtent()
    
    def GetResolution(self):
        if self.many_worlds is None:
            raise ValueError("ManyWorlds ownership has been transferred")
        return self.many_worlds.GetResolution()

    def GetOccupancy(self):
        if self.many_worlds is None:
            raise ValueError("ManyWorlds ownership has been transferred")
        return self.many_worlds.GetOccupancy()
    
    def GetNormal(self):
        if self.many_worlds is None:
            raise ValueError("ManyWorlds ownership has been transferred")
        return self.many_worlds.GetNormal()

    def UpdateNormals(self):
        if self.many_worlds is None:
            raise ValueError("ManyWorlds ownership has been transferred")
        self.many_worlds.UpdateNormals()

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