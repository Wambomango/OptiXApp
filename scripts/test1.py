import torch
import PyMWIR as mwir

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




renderer = mwir.InverseRenderer()

sender = mwir.Antenna((0, 0, 0), (0, 0, 0), (0.20, 0.10), 1E9)
receiver = mwir.Antenna((0, 0, 0), (0, 0, 0), (0.10, 0.10), 1E9)
signal = mwir.Signal((2 * torch.pi * 10e9, 2 * torch.pi * 10e9), 1)
mesh = mwir.Mesh()

scene = mwir.Scene(mesh, [sender], [receiver], signal)
many_worlds = mwir.ManyWorlds([0.05, -0.1, -0.1], [0.25, 0.1, 0.1], 0.001, 1)

occupancy = many_worlds.GetOccupancy()
with torch.no_grad():
    occupancy[:] = 0.1
    # occupancy[0:10, :, :] = 0.6

lr = 0.01
for i in range(1):
    with torch.no_grad():
        occupancy = many_worlds.GetOccupancy()
        occupancy.requires_grad = True
        occupancy.grad = None

    many_worlds.UpdateNormal()
    mesh.FromManyWorlds(many_worlds, 0.5)

    result = renderer.Render(scene, many_worlds, None)

    loss = result.abs().mean()
    loss.backward()

    # with torch.no_grad():
    #     occupancy = many_worlds.GetOccupancy()
    #     occupancy += lr * occupancy.grad




mesh.FromManyWorlds(many_worlds, 0.5)
mesh.View()











