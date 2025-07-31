import torch
import PyMWIR as mwir
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D








sender = mwir.Antenna((0, 0, 0), (0, 0, 0), (0.10, 0.10), 1E9)
receiver = mwir.Antenna((0, 0, 0), (0, 0, 0), (0.10, 0.10), 1E9)
signal = mwir.Signal((2 * torch.pi * 10e9, 2 * torch.pi * 10e9), 1)
scene = mwir.Scene(None, [sender], [receiver], signal)

many_worlds = mwir.ManyWorlds([[0.05, 0.25], [-0.2, 0.2], [-0.2, 0.2]], 0.001)

renderer = mwir.InverseRenderer()
renderer.Render(scene)