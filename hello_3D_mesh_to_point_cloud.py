#!/usr/bin/env python
"""
   Sample a 3D mesh to create a 3D point cloud
"""

#%% --------------------------------------------------------------------
# Prerequisites
import os
import sys
import copy
import numpy as np
import open3d as o3d


#%% ---------------------------------------------------------------------
# Apply Noise to the point cloud
def apply_noise(pcd, mu, sigma):
    """
    Applies Gaussian noise to the points of an Open3D PointCloud.

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        mu (float): The mean of the Gaussian noise distribution (typically 0).
        sigma (float): The standard deviation (intensity) of the noise.

    Returns:
        o3d.geometry.PointCloud: A new point cloud with added noise.
    """
    noisy_pcd = copy.deepcopy(pcd)

    points = np.asarray(noisy_pcd.points)
    # Generate noise with the same shape as the points array
    noise = np.random.normal(mu, sigma, size=points.shape)
    # Add the noise to the points
    points += noise
    # Update the point cloud points
    noisy_pcd.points = o3d.utility.Vector3dVector(points)

    return noisy_pcd


#%% -------------------------------------------------------------------
# Simulate Spherical Occlusion
def simulate_spherical_occlusion(pcd, center, radius):
    """
    Removes points within a sphere to simulate a round occlusion.
    """
    points = np.asarray(pcd.points)
    center = np.array(center)

    # Calculate Euclidean distance from every point to the center of the occlusion
    # np.linalg.norm computes the L2 norm (Euclidean distance) along the x, y, z axes
    distances = np.linalg.norm(points - center, axis=1)

    # Create a mask to select only the points *outside* the occlusion radius
    # The tilde (~) inverts the boolean mask (True becomes False, and vice versa)
    points_outside_occlusion = points[distances >= radius]

    # Create a new point cloud with the remaining points
    pcd_occluded = o3d.geometry.PointCloud()
    pcd_occluded.points = o3d.utility.Vector3dVector(points_outside_occlusion)
    
    return pcd_occluded


#%% --------------------------------------------------------------------
# Read and Visualize mesh with Open3D


MESH_PATH = "meshes_private/Toyota_FJ_Cruiser.obj"

if not os.path.exists(MESH_PATH):
    print(f"*** ERROR: File not found '{MESH_PATH}'")
    sys.exit(1)

# Read the mesh from the OBJ file
mesh = o3d.io.read_triangle_mesh(MESH_PATH)

if mesh.is_empty():
    print(f"*** ERROR: Failed to load mesh '{MESH_PATH}'")
    sys.exit(1)

# Optional: Calculate normals for better shading (if not present in the OBJ)
mesh.compute_vertex_normals()

# Visualize the mesh
print(f"Visualizing mesh: '{MESH_PATH}'")
o3d.visualization.draw_geometries([mesh], 
                                window_name="Open3D OBJ Visualizer",
                                width=800, height=600,
                                left=50, top=50)


#%% --------------------------------------------------------------------
# Sample points with uniform sampling and visualize point cloud
# Samples random points uniformly from triangle surfaces.

#nr_points_arr = [50000, 100000, 150000, 200000]
#nr_points_arr = [25000]
#nr_points_arr = [75000]
nr_points_arr = [40000]


# Sample points
for nr_points in nr_points_arr:
    point_cloud = mesh.sample_points_uniformly(number_of_points=nr_points)
    o3d.io.write_point_cloud(f"results/pc_uniform_{nr_points}_pts.ply", point_cloud)
    o3d.visualization.draw_geometries([point_cloud])


#%% --------------------------------------------------------------------
# Sample points with Poisson-disk sampling sampling and visualize point cloud
# Creates evenly spaced points simulating LiDAR uniformity

#nr_points_arr = [50000, 100000, 150000, 200000]
#nr_points_arr = [25000]
#nr_points_arr = [75000]
nr_points_arr = [40000]

for nr_points in nr_points_arr:
    point_cloud = mesh.sample_points_poisson_disk(
        number_of_points=nr_points,
        init_factor=5  # helps distribute points more evenly
    )

    o3d.io.write_point_cloud(f"results/pc_poisson_{nr_points}_pts.ply", point_cloud)
    o3d.visualization.draw_geometries([point_cloud])


#%% --------------------------------------------------------------------
# Visualize from specific view with a single color

COLOR = [1.0, 0.0, 0.0]

point_cloud = o3d.io.read_point_cloud("results/pointcloud_poisson.ply")

# Assign the same color to all points
# o3d.utility.Vector3dVector() expects a list of colors, one for each point.
# np.tile replicates the single color array for the total number of points.
point_cloud.colors = o3d.utility.Vector3dVector(np.tile(COLOR, (len(point_cloud.points), 1)))

"""
view_params = {
    "zoom": 0.5,
    "front": [0.0, 0.0, 1.0],  # Vector pointing from camera to origin (e.g., positive X direction)
    "lookat": [0.0, 0.0, 0.0], # The center of the object you're looking at
    "up": [0.0, 1.0, 0.0]      # Up direction of the camera (e.g., positive Z is up)
}
"""

view_params = {
    "zoom": 0.7,
	"front": [ -0.84312106568777812, 0.24138315335217866, 0.48050082400685235 ],
	"lookat":
			[
				-1.2993812561035156e-05,
				0.00069500505924224854,
				0.00031951069831848145
			],
	"up" : [ 0.17509471871899762, 0.96812265919773122, -0.17910989985098258 ]
}

o3d.io.write_point_cloud("results/pointcloud_poisson_angle.ply", point_cloud)

# Visualize the point cloud with the specific viewpoint
o3d.visualization.draw_geometries([point_cloud], **view_params)


#%% --------------------------------------------------------------------
# Occusion Removal from a given viewpoint

point_cloud = o3d.io.read_point_cloud("results/pc_poisson_50000_pts.ply")

# LiDAR/camera position in world coordinates
sensor_origin = np.array([0.0, 0.0, 1.5])  # x, y, z in meters
#sensor_origin = np.array([-1.0, 0.0, 1.5])  # x, y, z in meters

# Pick a radius that covers the scene (roughly max distance from sensor)
points = np.asarray(point_cloud.points)
distances = np.linalg.norm(points - sensor_origin, axis=1)
#radius = distances.max() * 1.1
radius = distances.max() * 25.0

# Visibility filtering
_, idx = point_cloud.hidden_point_removal(sensor_origin, radius)
visible_pcd = point_cloud.select_by_index(idx)

# Save / visualize
o3d.io.write_point_cloud("results/pc_poisson_50000_pts_visible.ply", visible_pcd)
o3d.visualization.draw_geometries([visible_pcd])



#%% ---------------------------------------------------------------------
# Load the point cloud (your existing code)
#point_cloud = o3d.io.read_point_cloud("results/pc_poisson_25000_pts.ply")
#point_cloud = o3d.io.read_point_cloud("results/pc_poisson_50000_pts.ply")
#point_cloud = o3d.io.read_point_cloud("results/pc_poisson_75000_pts.ply")
#point_cloud = o3d.io.read_point_cloud("results/pc_uniform_50000_pts.ply")
#point_cloud_orig = o3d.io.read_point_cloud("results/pc_uniform_50000_pts.ply")
#point_cloud = apply_noise(point_cloud_orig, 0, 0.005)

#point_cloud_orig = o3d.io.read_point_cloud("results/pc_uniform_50000_pts.ply")
#point_cloud_orig = o3d.io.read_point_cloud("results/pc_poisson_50000_pts.ply")
point_cloud_orig = o3d.io.read_point_cloud("results/pc_poisson_100000_pts.ply")

occl_center_1 = [-0.94, -0.17, 0.15] # Adjust position relative to your scene
occl_radius_1 = 0.15 
point_cloud_occ1 = simulate_spherical_occlusion(point_cloud_orig, occl_center_1, occl_radius_1)

occl_center_2 = [0.35, 0.3, 0.33] # Adjust position relative to your scene
occl_radius_2 = 0.1 # 0.2
point_cloud_occ2 = simulate_spherical_occlusion(point_cloud_occ1, occl_center_2, occl_radius_2)

occl_center_3 = [-0.83, 0.097, -0.065] # Adjust position relative to your scene
occl_radius_3 = 0.1 # 0.2
point_cloud_occ3 = simulate_spherical_occlusion(point_cloud_occ2, occl_center_3, occl_radius_3)


#point_cloud = apply_noise(point_cloud_occ3, 0, 0.005)
#point_cloud = point_cloud_occ3
point_cloud = point_cloud_orig


# Set Color
#COLOR = [1.0, 0.0, 0.0]
#point_cloud.colors = o3d.utility.Vector3dVector(np.tile(COLOR, (len(point_cloud.points), 1)))


view_params = {
    "zoom": 0.7,
    "front": [ -0.84312106568777812, 0.24138315335217866, 0.48050082400685235 ],
    "lookat": [ -1.2993812561035156e-05, 0.00069500505924224854, 0.00031951069831848145 ],
    "up" : [ 0.17509471871899762, 0.96812265919773122, -0.17910989985098258 ]
}

# --- Code to apply hidden point removal ---

## Calculate necessary parameters
# The 'front' vector is a unit vector pointing from the camera to the 'lookat' point
# We need the camera's position, so we estimate the distance to move back from the lookat point.

# 1. Estimate a suitable radius (a large value covering the scene)
# The diameter of the point cloud's bounding box is a good starting point.
diameter = np.linalg.norm(np.asarray(point_cloud.get_max_bound()) - np.asarray(point_cloud.get_min_bound()))
#radius = diameter * 100.0 # A large radius ensures points within the view are considered
radius = diameter * 200.0 # A large radius ensures points within the view are considered

# 2. Determine the camera's position (viewpoint origin)
# The 'front' vector points towards the scene. The camera position is behind the 'lookat' point
# along the negative 'front' direction. The 'zoom' factor affects the distance.
camera_distance = diameter / view_params["zoom"] # crude estimation of distance
#camera_position = np.asarray(view_params["lookat"]) - np.asarray(view_params["front"]) * camera_distance
camera_position = np.asarray(view_params["lookat"]) + np.asarray(view_params["front"]) * camera_distance

## Perform Hidden Point Removal (HPR)
# The function returns the visible point cloud (as a new point cloud object) and the depth map
_, pt_map = point_cloud.hidden_point_removal(camera_position, radius)

## Create a new point cloud containing only the visible points
# The 'pt_map' contains the indices of the visible points
visible_point_cloud = point_cloud.select_by_index(pt_map)

# Optional: Visualize the result
o3d.visualization.draw_geometries([visible_point_cloud], **view_params)
#o3d.visualization.draw_geometries_with_editing([visible_point_cloud])


# %%
