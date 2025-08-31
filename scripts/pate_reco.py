import torch
import PyMWIR as mwir
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time


plate_scale = 0.01
plate_distance = 5

mw_renderer = mwir.ManyWorldsRenderer()
renderer = mwir.Renderer()
y_axis = torch.linspace(-0.05, 0.05, 4)
z_axis = torch.linspace(-0.05, 0.05, 4)
antennas = []
for y in range(y_axis.shape[0]):
    for z in range(z_axis.shape[0]):
        antennas.append(mwir.Antenna((0, y_axis[y], z_axis[z]), (0, 0, 0), (0.2, 0.2), 1E9))
# antennas = [mwir.Antenna((0, 0, 0), (0, 0, 0), (0.2, 0.2), 1E10)]
signal = mwir.Signal((10e9, 20e9), 64)


mesh = mwir.Mesh(indices = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.uint32))
scene = mwir.Scene(mesh, antennas, antennas, signal)


opt_mesh = mwir.Mesh()
opt_scene = mwir.Scene(opt_mesh, antennas, antennas, signal)
many_worlds = mwir.ManyWorlds([plate_distance - 0.25, -0.25, -0.25], [plate_distance + 0.25, 0.25, 0.25], 0.001, 1)


with torch.no_grad():
    occupancy = many_worlds.GetOccupancy()
    print(occupancy.shape)
    occupancy[:] = 0.00
mesh_vertices = torch.tensor([[0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]], dtype=torch.float32) * plate_scale + torch.tensor([plate_distance, 0, 0], dtype=torch.float32)
mesh.SetVertices(mesh_vertices)



E_rx_ref = renderer.Render(scene)
E_rx = mw_renderer.Render(opt_scene, many_worlds)
loss = torch.sum(torch.abs(E_rx - E_rx_ref))
loss.backward()
torch.save(occupancy.grad, "/home/mario/Desktop/Masterarbeit/OptiXApp/scripts/occupancy_grad.pt")



occupancy_grad = torch.load("/home/mario/Desktop/Masterarbeit/OptiXApp/scripts/occupancy_grad.pt")
with torch.no_grad():
    occupancy = many_worlds.GetOccupancy()
    occupancy[:] = (torch.max(-occupancy_grad, torch.tensor(0)) / occupancy_grad.abs().max())
    occupancy[:] = torch.clip(occupancy, 0, 1)

mesh = mwir.Mesh()
mesh.FromManyWorlds(many_worlds, threshold = 0.3, add_bounds = True)
mesh.View()