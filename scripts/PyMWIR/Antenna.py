import torch
from . import PyBindMWIR

class Antenna:
    def __init__(self, position = (0, 0, 0), euler = (0, 0, 0), fov = (1, 1), ray_density = 1E9, impl = None):
        if impl is None:
            position = self.__SetPosition(position)
            euler = self.__SetEuler(euler)
            fov = self.__SetFOV(fov)
            ray_density = self.__SetRayDensity(ray_density)
            self.antenna = PyBindMWIR.Antenna(position, euler, fov, ray_density)
        elif type(impl) is PyBindMWIR.Antenna:
            self.antenna = impl
        else:
            raise TypeError("impl must be of type Antenna")

    def SetPosition(self, position):
        if self.antenna is None:
            raise ValueError("internal antenna instance has been moved.")
        position = self.__SetPosition(position)
        self.antenna.SetPosition(position)

    def SetEuler(self, euler):
        if self.antenna is None:
            raise ValueError("internal antenna instance has been moved.")
        euler = self.__SetEuler(euler)
        self.antenna.SetEuler(euler)

    def SetFOV(self, fov):
        if self.antenna is None:
            raise ValueError("internal antenna instance has been moved.")
        fov = self.__SetFOV(fov)
        self.antenna.SetFOV(fov)

    def SetRayDensity(self, ray_density):
        if self.antenna is None:
            raise ValueError("internal antenna instance has been moved.")
        ray_density = self.__SetRayDensity(ray_density)
        self.antenna.SetRayDensity(ray_density)

    def GetPosition(self):
        if self.antenna is None:
            raise ValueError("internal antenna instance has been moved.")
        return self.antenna.GetPosition()
    
    def GetOrientation(self):
        if self.antenna is None:
            raise ValueError("internal antenna instance has been moved.")
        return self.antenna.GetOrientation()
    
    def GetFOV(self):
        if self.antenna is None:
            raise ValueError("internal antenna instance has been moved.")
        return self.antenna.GetFOV()

    def GetRayDensity(self):
        if self.antenna is None:
            raise ValueError("internal antenna instance has been moved.")
        return self.antenna.GetRayDensity()

    def __SetPosition(self, position):
        if(type(position) is torch.Tensor):
            if(len(position.shape) != 1 or position.shape[0] != 3):
                raise ValueError("position must be a tensor of shape [3]")
        elif (type(position) is list or type(position) is tuple):
            if(len(position) != 3):
                raise ValueError("position must be a list or tuple of length 3")
        else:
            raise TypeError("position must be a torch.Tensor, list, or tuple")
        return torch.tensor(position, dtype=torch.float32)

    def __SetEuler(self, euler):
        if(type(euler) is torch.Tensor):
            if(len(euler.shape) != 1 or euler.shape[0] != 3):
                raise ValueError("euler must be a tensor of shape [3]")
        elif (type(euler) is list or type(euler) is tuple):
            if(len(euler) != 3):
                raise ValueError("euler must be a list or tuple of length 3")
        else:
            raise TypeError("euler must be a torch.Tensor, list, or tuple")
        return torch.tensor(euler, dtype=torch.float32)

    def __SetFOV(self, fov):
        if(type(fov) is torch.Tensor):
            if(len(fov.shape) != 1 or fov.shape[0] != 2):
                raise ValueError("fov must be a tensor of shape [2]")
        elif (type(fov) is list or type(fov) is tuple):
            if(len(fov) != 2):
                raise ValueError("fov must be a list or tuple of length 2")
        else:
            raise TypeError("fov must be a torch.Tensor, list, or tuple")
        return torch.tensor(fov, dtype=torch.float32)

    def __SetRayDensity(self, ray_density):
        if(type(ray_density) is not float and type(ray_density) is not int):
            raise TypeError("ray_density must be a float or int")
        return torch.tensor((ray_density, ), dtype=torch.float32)