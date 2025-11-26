#!/usr/bin/env python
"""
   Sample a 3D mesh to create a 3D point cloud
"""

#%% --------------------------------------------------------------------
# Prerequisites
import os
import sys
import numpy as np
import open3d as o3d


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

# Sample ~50k points
nr_points = 50000
point_cloud = mesh.sample_points_uniformly(number_of_points=nr_points)

o3d.io.write_point_cloud("pointcloud_uniform.ply", point_cloud)
o3d.visualization.draw_geometries([point_cloud])


#%% --------------------------------------------------------------------
# Sample points with Poisson-disk sampling sampling and visualize point cloud
# Creates evenly spaced points simulating LiDAR uniformity

point_cloud = mesh.sample_points_poisson_disk(
    number_of_points=nr_points,
    init_factor=5  # helps distribute points more evenly
)

o3d.io.write_point_cloud("pointcloud_poisson.ply", point_cloud)
o3d.visualization.draw_geometries([point_cloud])


#%% --------------------------------------------------------------------
# Visualize from specific view with a single color

COLOR = [1.0, 0.0, 0.0]

point_cloud = o3d.io.read_point_cloud("pointcloud_poisson.ply")

# Assign the same color to all points
# o3d.utility.Vector3dVector() expects a list of colors, one for each point.
# np.tile replicates the single color array for the total number of points.
point_cloud.colors = o3d.utility.Vector3dVector(np.tile(COLOR, (len(point_cloud.points), 1)))

view_params = {
    "zoom": 0.5,
    "front": [0.0, 0.0, 1.0],  # Vector pointing from camera to origin (e.g., positive X direction)
    "lookat": [0.0, 0.0, 0.0], # The center of the object you're looking at
    "up": [0.0, 1.0, 0.0]      # Up direction of the camera (e.g., positive Z is up)
}

# Visualize the point cloud with the specific viewpoint
o3d.visualization.draw_geometries([point_cloud], **view_params)


#%% --------------------------------------------------------------------
# Occusion Removal from a given viewpoint

# pcd: existing point cloud (e.g. from mesh.sample_points_poisson_disk)
pcd = o3d.io.read_point_cloud("pointcloud_poisson.ply")

# LiDAR/camera position in world coordinates
sensor_origin = np.array([0.0, 0.0, 1.5])  # x, y, z in meters

# Pick a radius that covers the scene (roughly max distance from sensor)
points = np.asarray(pcd.points)
distances = np.linalg.norm(points - sensor_origin, axis=1)
radius = distances.max() * 1.1

# Visibility filtering
_, idx = pcd.hidden_point_removal(sensor_origin, radius)
visible_pcd = pcd.select_by_index(idx)

# Save / visualize
o3d.io.write_point_cloud("scene_visible_from_sensor.ply", visible_pcd)
o3d.visualization.draw_geometries([visible_pcd])


#%% --------------------------------------------------------------------
# Raycasting

# 1) Load triangle mesh
print("1. Load triangle mesh")
mesh_legacy = o3d.io.read_triangle_mesh(MESH_PATH)
mesh_legacy.compute_vertex_normals()

# Convert to tensor mesh for raycasting
mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)

scene = o3d.t.geometry.RaycastingScene()
_ = scene.add_triangles(mesh)

# 2) LiDAR pose
print("2. Lidar Pose")
origin = np.array([0.0, 0.0, 1.5], dtype=np.float32)  # LiDAR center in world frame

# 3) Define LiDAR scanning pattern (example: 64 beams, 0.2° azimuth)
print("3. Define LiDAR scanning pattern (example: 64 beams, 0.2° azimuth)")

n_vert = 64
n_horiz = 1800   # 360 / 0.2

vert_min_deg, vert_max_deg = -24.8, 2.0  # e.g. Velodyne-ish
vert_angles = np.linspace(np.radians(vert_min_deg),
                          np.radians(vert_max_deg),
                          n_vert)

horiz_angles = np.linspace(-np.pi, np.pi, n_horiz, endpoint=False)

dirs = []
origins = []

for v in vert_angles:
    for h in horiz_angles:
        # Spherical → Cartesian
        dx = np.cos(v) * np.cos(h)
        dy = np.cos(v) * np.sin(h)
        dz = np.sin(v)
        d = np.array([dx, dy, dz], dtype=np.float32)

        dirs.append(d)
        origins.append(origin)

origins = np.stack(origins, axis=0)
dirs = np.stack(dirs, axis=0)

# 4) Build ray tensor: [x, y, z, dx, dy, dz]
print("4. Build ray tensor: [x, y, z, dx, dy, dz]")
rays = np.concatenate([origins, dirs], axis=1).astype(np.float32)
rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

# 5) Cast rays
print("5. Cast rays")
ans = scene.cast_rays(rays)

# ans["t_hit"] contains distance along each ray; inf if no hit
t_hit = ans["t_hit"].numpy().reshape(-1, 1)
mask = np.isfinite(t_hit).flatten()

t_hit_valid = t_hit[mask]
rays_valid = rays.numpy()[mask]

# 6) Compute hit points: P = O + t * D
print("6. Compute hit points: P = O + t * D")
origins_valid = rays_valid[:, :3]
dirs_valid = rays_valid[:, 3:]
points = origins_valid + dirs_valid * t_hit_valid

# 7) Make point cloud
print("7. Make point cloud")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

o3d.io.write_point_cloud("simulated_lidar_scan.ply", pcd)
o3d.visualization.draw_geometries([pcd])

