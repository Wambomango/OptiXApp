import torch
import PyMWIR as mwir

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




renderer = mwir.InverseRenderer()

sender = mwir.Antenna((0, 0, 0), (0, 0, 0), (0.10, 0.10), 1E9)
receiver = mwir.Antenna((0, 0, 0), (0, 0, 0), (0.10, 0.10), 1E9)
signal = mwir.Signal((2 * torch.pi * 10e9, 2 * torch.pi * 10e9), 1)
mesh = mwir.Mesh()

scene = mwir.Scene(mesh, [sender], [receiver], signal)
many_worlds = mwir.ManyWorlds([0.05, -0.1, -0.1], [0.25, 0.1, 0.1], 0.001)

occupancy = many_worlds.GetOccupancy()

with torch.no_grad():
    # occupancy[:] = torch.rand(occupancy.shape, dtype=torch.float32, device="cuda") * 1
    occupancy[60:120, 60:120, 60:120] = 1.0

mesh.FromManyWorlds(many_worlds, 0.5)
mesh.View()



# mesh.ToObj("/tmp/test.obj")







# plt.imshow(occupancy.detach().cpu().numpy()[:,0,:], cmap='gray')
# plt.colorbar()
# plt.show()





# renderer.Render(scene, many_worlds, None)











# lr = 0.01

# with torch.no_grad():
#     occupancy = many_worlds.GetOccupancy()
#     occupancy.requires_grad = True
#     occupancy.grad = None

# many_worlds.UpdateNormals()


# # scene.SetMesh(many_worlds.GenerateMesh())

# result = renderer.Render(scene, many_worlds, None)

# loss = result
# loss.backward()

# with torch.no_grad():
#     occupancy = many_worlds.GetOccupancy()
#     occupancy += lr * occupancy.grad




# a = torch.randn((100, 100), dtype=torch.float32)
# a_grad = t

# plt.imshow(a.cpu().detach().numpy(), cmap='gray')


# plt.colorbar()
# plt.show()





