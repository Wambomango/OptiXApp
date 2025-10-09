import torch
import PyMWIR as mwir
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time


plate_scale = 0.05
plate_distance = 0.3
many_worlds_scale = 0.1
many_worlds_resolution = 0.0008


mw_renderer = mwir.ManyWorldsRenderer()
renderer = mwir.Renderer()




n_axis = 1
n_total = n_axis**2
density = 1E9 / n_total
distance = 0.02
axis = torch.arange(n_axis) * distance
axis -= axis.max() / 2
antennas = []
for y in range(n_axis):
    for z in range(n_axis):
        antennas.append(mwir.Antenna((0, axis[y], axis[z]), (0, 0, 0), (0.5, 0.5), density))
signal = mwir.Signal((10e9, 20e9), 2)


mesh = mwir.Mesh(indices = torch.tensor([[0, 1, 2], [1, 3, 2]], dtype=torch.uint32))
scene = mwir.Scene(mesh, antennas, antennas, signal)

opt_mesh = mwir.Mesh()
opt_scene = mwir.Scene(opt_mesh, antennas, antennas, signal)
many_worlds = mwir.ManyWorlds([plate_distance - many_worlds_scale, -many_worlds_scale, -many_worlds_scale], [plate_distance + many_worlds_scale, many_worlds_scale, many_worlds_scale], many_worlds_resolution, 1)


with torch.no_grad():
    occupancy = many_worlds.GetOccupancy()
    occupancy[:] = 0.01
mesh_vertices = torch.tensor([[0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]], dtype=torch.float32) * plate_scale + torch.tensor([plate_distance, 0, 0], dtype=torch.float32)
mesh.SetVertices(mesh_vertices)



for i in range(10):
    t_start = time.time()
    E_rx_ref = renderer.Render(scene)
    E_rx = mw_renderer.Render(opt_scene, many_worlds)
    loss = torch.sum(torch.abs(E_rx - E_rx_ref))
    loss.backward()
    t_stop = time.time()
    print("Render time: ", t_stop - t_start)


    with torch.no_grad():
        occupancy = many_worlds.GetOccupancy()
        occupancy[:] += (-occupancy.grad / occupancy.grad.abs().max()) * 0.2
        occupancy[:] += torch.clip(occupancy, 0, 1)
        occupancy.grad = None
    opt_mesh.FromManyWorlds(many_worlds, threshold = 0.5, add_bounds = False)


opt_mesh.View()








