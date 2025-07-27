import torch
import PyMWIR as mwir

plate_scale = 0.5
plate_distance = 10
vertices = torch.tensor([[0, -1, -1], [0, 1, 1], [0, 1, -1], [0, -1, -1], [0, -1, 1], [0, 1, 1]], dtype=torch.float32) * plate_scale + torch.tensor([plate_distance, 0, 0], dtype=torch.float32)
mesh = mwir.Mesh(vertices)

sender = mwir.Antenna((0, 0, 0), (0, 0, 0), (0.10, 0.10), 1E9)
receiver1 = mwir.Antenna((0, 0, 0), (0, 0, 0), (0.10, 0.10), 1E9)
receiver2 = mwir.Antenna((0, 0, 0), (0, 0, 0), (0.10, 0.10), 1E9)

signal = mwir.Signal((10e9, 20e9), 10)

scene = mwir.Scene(mesh, [sender], [receiver1, receiver2], signal)
renderer = mwir.Renderer(scene)

result = renderer.Render()

print(result)
