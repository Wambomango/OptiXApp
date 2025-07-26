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





# import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt


# fovx = 1.0  # Field of view in radians
# fovy = 1.0  # Field of view in radians

# u = np.random.uniform(0, 1, 100000)
# v = np.random.uniform(0, 1, 100000)
# azimuth = fovx * (u - 0.5)
# elevation = np.arcsin(np.sin(fovy / 2) * (2 * v - 1))


# # Convert spherical coordinates (azimuth, elevation) to Cartesian coordinates
# x = np.cos(elevation) * np.cos(azimuth)
# y = np.cos(elevation) * np.sin(azimuth)
# z = np.sin(elevation)

# # az = np.arctan2(y, x)
# # el = np.arctan2(z, np.sqrt(x**2 + y**2))
# # print(az.max(), az.min())
# # print(el.max(), el.min())





# # Plot the points in 3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z, s=1)

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()