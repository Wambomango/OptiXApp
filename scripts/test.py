import torch
import PyMWIR as mwir


vertices = torch.tensor([[0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, -1], [0, 1, 1], [0, -1, 1]], dtype=torch.float32) * 0.2 + 100
mesh = mwir.Mesh(vertices)

sender = mwir.Antenna((0, 0, 0), (0, 0, 0), (0.10, 0.10), 1E9)
receiver1 = mwir.Antenna((0, 0, 0), (0, 0, 0), (0.10, 0.10), 1E9)
receiver2 = mwir.Antenna((0, 0, 0), (0, 0, 0), (0.10, 0.10), 1E9)

signal = mwir.Signal((10e9, 20e9), 100)

scene = mwir.Scene(mesh, [sender], [receiver1, receiver2], signal)
renderer = mwir.Renderer(scene)

result = renderer.Render()



# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# def sample_sphere(N, azimuth_range, elevation_range):
#     phi_min, phi_max = azimuth_range
#     theta_min, theta_max = elevation_range

#     # Sample azimuth φ uniformly
#     phi = np.random.uniform(phi_min, phi_max, N)

#     # Sample cos(θ) uniformly for correct area weighting
#     cos_theta_min = np.sin(theta_min)
#     cos_theta_max = np.sin(theta_max)
#     cos_theta = np.random.uniform(cos_theta_max, cos_theta_min, N)
#     theta = np.arcsin(cos_theta)

#     # Convert spherical to Cartesian coordinates (assuming unit sphere)
#     x = np.cos(theta) * np.cos(phi)
#     y = np.cos(theta) * np.sin(phi)
#     z = np.sin(theta)

#     return np.stack([x, y, z], axis=1)




# import matplotlib.pyplot as plt

# # Sample 1000 points on the sphere
# N = 1000
# azimuth_range = (-1, 1)
# elevation_range = (-1,1)
# points = sample_sphere(N, azimuth_range, elevation_range)

# # Make axes have equal scale
# max_range = (points.max(axis=0) - points.min(axis=0)).max() / 2.0
# mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
# mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
# mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

# ax_limits = [
#     (mid_x - max_range, mid_x + max_range),
#     (mid_y - max_range, mid_y + max_range),
#     (mid_z - max_range, mid_z + max_range)
# ]


# # Plot in 3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5)
# ax.set_xlim(ax_limits[0])
# ax.set_ylim(ax_limits[1])
# ax.set_zlim(ax_limits[2])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('Sampled Points on Sphere')
# plt.show()