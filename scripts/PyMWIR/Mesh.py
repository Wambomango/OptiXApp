import torch
from traitlets import Container
from . import PyBindMWIR
from .ManyWorlds import ManyWorlds
from pytorch3d.ops import marching_cubes  
import os
import subprocess
from tqdm import tqdm
import itertools

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

    def FromManyWorlds(self, many_worlds, threshold = 0.5, add_bounds = False):
        if self.mesh is None:
            raise ValueError("Mesh ownership has been transferred")
        if type(many_worlds) is not ManyWorlds:
            raise TypeError("many_worlds must be of type ManyWorlds")

        occupancy = many_worlds.GetOccupancy().cuda()
        padded_occupancy = torch.nn.functional.pad(occupancy, (1, 1, 1, 1, 1, 1), mode='constant', value=0)
        vertices, indices = marching_cubes.marching_cubes(padded_occupancy[None, :, :, :], threshold)
        if type(vertices[0]) is torch.Tensor and type(indices[0]) is torch.Tensor:
            vertices = vertices[0]
            indices = indices[0].to(torch.uint32)
            tmp = vertices[:, 2].clone()
            vertices[:, 2] = vertices[:, 0]
            vertices[:, 0] = tmp
            scale_correction = (torch.tensor(padded_occupancy.shape)) / (torch.tensor(occupancy.shape))
            vertices[:, 0] = vertices[:, 0] * scale_correction[0]
            vertices[:, 1] = vertices[:, 1] * scale_correction[1]
            vertices[:, 2] = vertices[:, 2] * scale_correction[2]
            extent_min = many_worlds.GetMin().cuda()
            extent_max = many_worlds.GetMax().cuda()
            vertices = (vertices + 1.0) * 0.5 * (extent_max - extent_min) + extent_min
        else:
            vertices = torch.zeros((0, 3), dtype=torch.float32, device="cuda")
            indices = torch.zeros((0, 3), dtype=torch.uint32, device="cuda")



        if add_bounds:
            min_bounds = many_worlds.GetMin()
            max_bounds = many_worlds.GetMax()
            # 8 corners of the box
            corners = torch.stack([
                torch.tensor([min_bounds[0], min_bounds[1], min_bounds[2]]),
                torch.tensor([max_bounds[0], min_bounds[1], min_bounds[2]]),
                torch.tensor([max_bounds[0], max_bounds[1], min_bounds[2]]),
                torch.tensor([min_bounds[0], max_bounds[1], min_bounds[2]]),
                torch.tensor([min_bounds[0], min_bounds[1], max_bounds[2]]),
                torch.tensor([max_bounds[0], min_bounds[1], max_bounds[2]]),
                torch.tensor([max_bounds[0], max_bounds[1], max_bounds[2]]),
                torch.tensor([min_bounds[0], max_bounds[1], max_bounds[2]]),
            ], dim=0).to(torch.float32)

            # 12 edges as pairs of corner indices
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
                (4, 5), (5, 6), (6, 7), (7, 4),  # top face
                (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
            ]

            # For each edge, define the two adjacent face normals
            edge_normals = [
                [(0, 0, -1), (0, -1, 0)],  # (0,1)
                [(0, 0, -1), (1, 0, 0)],   # (1,2)
                [(0, 0, -1), (0, 1, 0)],   # (2,3)
                [(0, 0, -1), (-1, 0, 0)],  # (3,0)
                [(0, 0, 1), (0, -1, 0)],   # (4,5)
                [(0, 0, 1), (1, 0, 0)],    # (5,6)
                [(0, 0, 1), (0, 1, 0)],    # (6,7)
                [(0, 0, 1), (-1, 0, 0)],   # (7,4)
                [(-1, 0, 0), (0, -1, 0)],  # (0,4)
                [(1, 0, 0), (0, -1, 0)],   # (1,5)
                [(1, 0, 0), (0, 1, 0)],    # (2,6)
                [(-1, 0, 0), (0, 1, 0)],   # (3,7)
            ]

            thickness = 0.01 * torch.norm(max_bounds - min_bounds)
            bounds_vertices = []
            bounds_indices = []
            vtx_count = 0

            for i, (a, b) in enumerate(edges):
                p0 = corners[a]
                p1 = corners[b]
                edge_dir = (p1 - p0)
                edge_dir = edge_dir / (torch.norm(edge_dir) + 1e-8)

                # For both adjacent face normals
                for normal in edge_normals[i]:
                    normal_vec = torch.tensor(normal, dtype=torch.float32)
                    perp = torch.cross(edge_dir, normal_vec)
                    perp = perp / (torch.norm(perp) + 1e-8) * thickness

                    # Four vertices for the rectangle
                    v0 = p0 + perp
                    v1 = p0 - perp
                    v2 = p1 + perp
                    v3 = p1 - perp

                    bounds_vertices.extend([v0, v1, v2, v3])
                    # Two triangles per rectangle
                    bounds_indices.append([vtx_count + 0, vtx_count + 1, vtx_count + 2])
                    bounds_indices.append([vtx_count + 1, vtx_count + 3, vtx_count + 2])
                    vtx_count += 4

            bounds_vertices = torch.stack(bounds_vertices, dim=0).cuda()
            bounds_indices = (torch.tensor(bounds_indices) + vertices.shape[0]).to(torch.uint32).cuda()

            vertices = torch.cat([vertices, bounds_vertices], dim=0)
            indices = torch.cat([indices, bounds_indices], dim=0)


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