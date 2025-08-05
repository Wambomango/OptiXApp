import torch
from traitlets import Container
from . import PyBindMWIR
from .ManyWorlds import ManyWorlds
from pytorch3d.ops import marching_cubes  
import os
import subprocess
from tqdm import tqdm

class Mesh:
    def __init__(self, vertices = torch.zeros((0, 3), dtype=torch.float32), indices = torch.zeros((0, 3), dtype=torch.uint32), impl = None):
        if impl is None:
            vertices = self.__SetVertices(vertices)
            indices = self.__SetIndices(indices)
            self.mesh = PyBindMWIR.Mesh(vertices, indices)
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

    def SetIndices(self, indices):
        if self.mesh is None:
            raise ValueError("Mesh ownership has been transferred")
        indices = self.__SetIndices(indices)
        self.mesh.SetIndices(indices)

    def GetIndices(self):
        if self.mesh is None:
            raise ValueError("Mesh ownership has been transferred")
        return self.mesh.GetIndices()

    def FromManyWorlds(self, many_worlds, threshold = 0.5):
        if self.mesh is None:
            raise ValueError("Mesh ownership has been transferred")
        if type(many_worlds) is not ManyWorlds:
            raise TypeError("many_worlds must be of type ManyWorlds")

        occupancy = many_worlds.GetOccupancy()
        extent_min = many_worlds.GetMin().cuda()
        extent_max = many_worlds.GetMax().cuda()
        vertices, indices = marching_cubes.marching_cubes(occupancy[None, :, :, :], threshold)
        if type(vertices[0]) is torch.Tensor and type(indices[0]) is torch.Tensor:
            vertices = vertices[0]
            indices = indices[0].to(torch.uint32)
            print(vertices.shape)
            tmp = vertices[:, 2].clone()
            vertices[:, 2] = vertices[:, 0]
            vertices[:, 0] = tmp
            vertices = (vertices + 1.0) * 0.5 * (extent_max - extent_min) + extent_min
            self.mesh.SetVertices(vertices)
            self.mesh.SetIndices(indices)
        else:
            vertices = torch.zeros((0, 3), dtype=torch.float32, device="cuda")
            indices = torch.zeros((0, 3), dtype=torch.uint32, device="cuda")
            self.mesh.SetVertices(vertices)
            self.mesh.SetIndices(indices)


    def ToObj(self, filename):
        with open(filename, 'w') as f:
            # Write vertices
            with torch.no_grad():
                vertices = self.mesh.GetVertices().detach().cpu()
                indices = self.mesh.GetIndices().detach().cpu()
                f.writelines(
                        f"v {v[0].item()} {v[1].item()} {v[2].item()}\n"
                        for v in tqdm(vertices)
                )
                f.writelines(
                    f"f {idx[0].item()+1} {idx[1].item()+1} {idx[2].item()+1}\n"
                    for idx in tqdm(indices)
                )

    def View(self):
        if self.mesh is None:
            raise ValueError("Mesh ownership has been transferred")
        mwir_dir = os.path.dirname(os.path.abspath(__file__))

        viewer_path = os.path.join(mwir_dir, "Viewer")
        if not os.path.exists(viewer_path):
            raise FileNotFoundError(f"Viewer executable not found at {viewer_path}")

        vertices = self.mesh.GetVertices().detach().cpu()
        indices = self.mesh.GetIndices().detach().cpu().to(torch.int32)
        class Container(torch.nn.Module):
            def __init__(self, my_values):
                super().__init__()
                for key in my_values:
                    setattr(self, key, my_values[key])
                    
        container = torch.jit.script(Container({"vertices": vertices, "indices": indices}))
        container.save("/tmp/pymwir_mesh.pt")
        subprocess.run([viewer_path, "/tmp/pymwir_mesh.pt"])


    def __SetVertices(self, vertices):
        if(type(vertices) is not torch.Tensor):
            raise TypeError("vertices must be a torch.Tensor")
        if(len(vertices.shape) != 2 or vertices.shape[1] != 3):
            raise ValueError("vertices must be a tensor of shape [N, 3]")
        return vertices

    def __SetIndices(self, indices):
        if(type(indices) is not torch.Tensor):
            raise TypeError("indices must be a torch.Tensor")
        if(len(indices.shape) != 2 or indices.shape[1] != 3):
            raise ValueError("indices must be a tensor of shape [N, 3]")
        return indices