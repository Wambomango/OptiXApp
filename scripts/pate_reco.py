import torch
import PyMWIR as mwir
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


plate_scale = 0.5
plate_distance = 5

mw_renderer = mwir.ManyWorldsRenderer()
renderer = mwir.Renderer()

sender = mwir.Antenna((0, 0, 0), (0, 0, 0), (0.2, 0.2), 1E10)
receiver = mwir.Antenna((0, 0, 0), (0, 0, 0), (0.10, 0.10), 1E10)
signal = mwir.Signal((10e9, 10e9), 1)
mesh = mwir.Mesh(indices = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.uint32))
scene = mwir.Scene(mesh, [sender], [receiver], signal)
many_worlds = mwir.ManyWorlds([plate_distance - 0.5, -0.5, -0.5], [plate_distance + 0.5, 0.5, 0.5], 0.01, 2)



with torch.no_grad():
    occupancy = many_worlds.GetOccupancy()
    occupancy[:] = 0.0
    many_worlds.UpdateNormal()

mesh_vertices = torch.tensor([[0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]], dtype=torch.float32) * plate_scale + torch.tensor([plate_distance, 0, 0], dtype=torch.float32)
mesh.SetVertices(mesh_vertices)
E_rx_ref = renderer.Render(scene)
E_rx = mw_renderer.Render(scene, many_worlds)

print(E_rx_ref)
print(E_rx)


# loss = torch.sum(torch.abs(E_rx - E_rx_ref))
# loss.backward()
# print(occupancy.grad.shape)
