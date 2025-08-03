import torch
from . import PyBindMWIR

class ManyWorlds:
    def __init__(self, min = torch.tensor([-0.1, -0.1, -0.1]), max = torch.tensor([0.1, 0.1, 0.1]), resolution = 0.001, impl = None):
        min = self.__SetMin(min)
        max = self.__SetMax(max)
        resolution = self.__SetResolution(resolution)
        self.many_worlds = PyBindMWIR.ManyWorlds(min, max, resolution)

    def SetMin(self, min):
        if self.many_worlds is None:
            raise ValueError("ManyWorlds ownership has been transferred")
        min = self.__SetMin(min)
        self.many_worlds.SetMin(min)

    def SetMax(self, max):
        if self.many_worlds is None:
            raise ValueError("ManyWorlds ownership has been transferred")
        max = self.__SetMax(max)
        self.many_worlds.SetMax(max)

    def SetResolution(self, resolution):
        if self.many_worlds is None:
            raise ValueError("ManyWorlds ownership has been transferred")
        resolution = self.__SetResolution(resolution)
        self.many_worlds.SetResolution(resolution)

    def GetMin(self):
        if self.many_worlds is None:
            raise ValueError("ManyWorlds ownership has been transferred")
        return self.many_worlds.GetMin()

    def GetMax(self):
        if self.many_worlds is None:
            raise ValueError("ManyWorlds ownership has been transferred")
        return self.many_worlds.GetMax()

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

    def __SetMin(self, min):
        if(type(min) is torch.Tensor):
            if(len(min.shape) != 1 or min.shape[0] != 3):
                raise ValueError("min must be a tensor of shape [3]")
        elif (type(min) is list or type(min) is tuple):
            if(len(min) != 3):
                raise ValueError("min must be a list or tuple of length 3")
        else:
            raise TypeError("min must be a torch.Tensor, list, or tuple")
        return torch.tensor(min, dtype=torch.float32)

    def __SetMax(self, max):
        if(type(max) is torch.Tensor):
            if(len(max.shape) != 1 or max.shape[0] != 3):
                raise ValueError("max must be a tensor of shape [3]")
        elif (type(max) is list or type(max) is tuple):
            if(len(max) != 3):
                raise ValueError("max must be a list or tuple of length 3")
        else:
            raise TypeError("max must be a torch.Tensor, list, or tuple")
        return torch.tensor(max, dtype=torch.float32)

    def __SetResolution(self, resolution):
        if(type(resolution) is not float and type(resolution) is not int):
            raise TypeError("resolution must be a float or int")
        if(resolution <= 0):
            raise ValueError("resolution must be greater than 0")
        return float(resolution)