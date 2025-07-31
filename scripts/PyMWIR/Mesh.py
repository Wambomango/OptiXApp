import torch
from . import PyBindMWIR

class Mesh:
    def __init__(self, vertices = torch.zeros((0, 3), dtype=torch.float32), impl = None):
        if impl is None:
            vertices = self.SetVertices(vertices)
            self.mesh = PyBindMWIR.Mesh(vertices)
        elif type(impl) is PyBindMWIR.Mesh:
            self.mesh = impl
        else:
            raise TypeError("impl must be of type Mesh")

    def SetVertices(self, vertices):
        if self.mesh is None:
            raise ValueError("Mesh ownership has been transferred")
        vertices = self.__SetVertices(vertices)
        self.mesh.SetVertices(vertices)
       
    def GetVertices(self):
        if self.mesh is None:
            raise ValueError("Mesh ownership has been transferred")
        return self.mesh.GetVertices()
    
    def __SetVertices(self, vertices):
        if(type(vertices) is not torch.Tensor):
            raise TypeError("vertices must be a torch.Tensor")
        if(len(vertices.shape) != 2 or vertices.shape[1] != 3 or (vertices.shape[0] % 3) != 0):
            raise ValueError("vertices must be a tensor of shape [3N, 3]")
        return vertices
