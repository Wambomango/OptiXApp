import torch
import PyMWIR as mwir
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



plate_scale = 0.5
plate_distance = 100

sender = mwir.Antenna((0, 0, 0), (0, 0, 0), (0.10, 0.10), 1E9)
receiver = mwir.Antenna((0, 0, 0), (0, 0, 0), (0.10, 0.10), 1E9)
signal = mwir.Signal((2 * torch.pi * 10e9, 2 * torch.pi * 10e9), 1)
scene = mwir.Scene(None, [sender], [receiver], signal)
renderer = mwir.InverseRenderer()

n_angles = 1000
angles = torch.linspace(-10, 10, n_angles) * torch.pi / 180
rcs = torch.zeros(n_angles, dtype=torch.float32)

for i in range(n_angles):

    mesh = scene.GetMesh()
    angle = angles[i]
    rotation_matrix = torch.tensor([[torch.cos(angle), -torch.sin(angle), 0],
                                        [torch.sin(angle), torch.cos(angle), 0],
                                        [0, 0, 1]], dtype=torch.float32)
    vertices = torch.tensor([[0, -1, -1], [0, 1, 1], [0, 1, -1], [0, -1, -1], [0, -1, 1], [0, 1, 1]], dtype=torch.float32) @ rotation_matrix.T * plate_scale + torch.tensor([plate_distance, 0, 0], dtype=torch.float32)
    mesh.SetVertices(vertices)
    scene.SetMesh(mesh)
 
    E_rx = renderer.Render(scene)
    rcs[i] = 10 * torch.log10(4 * torch.pi * plate_distance**2 * torch.linalg.vector_norm(E_rx, dim = 2)**2)


plt.plot(angles * 180 / torch.pi, rcs)
plt.xlabel('Angle (degrees)')
plt.ylabel('RCS (dBsm)')
plt.title('RCS vs Angle')
plt.grid()
plt.show()
