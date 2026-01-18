#!/usr/bin/env python
"""

Hello PyVista

"""


#%%---------------------------------------------------------------------------
# Prerequisites and Dependencies
import numpy as np
import pyvista as pv

# pip install trame trame-vtk trame-client trame-server pyvista[jupyter] 
# (trane in needed for interactive viewing within the notebook)


#%%---------------------------------------------------------------------------
# PyVista example Pointy cloud

from pyvista import examples

datasets = examples.download_lucy()
datasets.plot(smooth_shading=True, color='white')


#%%---------------------------------------------------------------------------
# PyVista example Lidar

dataset_lidar = examples.download_lidar()
print(f"Downloading complete. Downloaded {dataset_lidar.n_points} points")
print(f"Data type {type(dataset_lidar)}")

dataset_lidar.plot(point_size=2, render_points_as_spheres=True, color='white')


#%%---------------------------------------------------------------------------
# Visualize Random Point Cloud 

# Create sample point cloud data (1000 random 3D points)
points = np.random.rand(1000, 3) * 10  # Shape: (N, 3)

# Create a PyVista point cloud from the numpy array
pcloud = pv.PolyData(points)

# Basic visualization
pcloud.plot(point_size=10, render_points_as_spheres=True)


#%%---------------------------------------------------------------------------
# Adding scalar values (colors based on data)

# Create point cloud
points = np.random.rand(500, 3) * 10

# Create scalar values (e.g., height-based coloring)
scalars = points[:, 2]  # Color by Z coordinate

# Create PyVista object and add scalars
pcloud = pv.PolyData(points)
pcloud["elevation"] = scalars

# Plot with colormap
pcloud.plot(
    scalars="elevation",
    cmap="viridis",
    point_size=12,
    render_points_as_spheres=True,
    show_scalar_bar=True
)

#%%---------------------------------------------------------------------------
# Interactive plotter with more control

# Create point cloud
points = np.random.rand(2000, 3)
points[:, 2] *= 0.5  # Flatten Z axis

cloud = pv.PolyData(points)
cloud["values"] = np.linalg.norm(points, axis=1)  # Distance from origin

# Create plotter for more control
plotter = pv.Plotter()
plotter.add_mesh(
    cloud,
    scalars="values",
    cmap="plasma",
    point_size=8,
    render_points_as_spheres=True
)
plotter.add_axes()
plotter.show_grid()
plotter.show()



#%%---------------------------------------------------------------------------
# RGB Colors per point

points = np.random.rand(1000, 3) * 10
colors = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)

cloud = pv.PolyData(points)
cloud["RGB"] = colors

cloud.plot(
    scalars="RGB",
    rgb=True,
    point_size=10,
    render_points_as_spheres=True
)


#%%---------------------------------------------------------------------------
#


#%%---------------------------------------------------------------------------
#


#%%---------------------------------------------------------------------------
#


