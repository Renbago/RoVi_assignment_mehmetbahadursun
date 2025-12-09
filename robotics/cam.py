from PIL import Image
import numpy as np
import mujoco as mj
import open3d as o3d
import struct

def get_rgb(m, d, renderer):
    renderer.update_scene(d, camera="cam1")
    rgb_img = renderer.render()
    im = Image.fromarray(rgb_img)
    im.save("rgb_img.jpeg")

def get_depth(m, d, renderer):
    # Update the scene with the specified camera
    renderer.update_scene(d, camera="cam1")    
    renderer.enable_depth_rendering()
    depth_img = renderer.render()
    renderer.disable_depth_rendering()

     # Convert depth to grayscale (0-255) for visualization
    # Depth values: 1.0 = near, 0.0 = far
    depth_visual = ((1.0 - depth_img) * 255).astype(np.uint8)

    im = Image.fromarray(depth_visual)
    im.save("depth_img.jpeg")
    
def get_pointcloud(model, data, renderer, camera_name="cam1"):
    """Simplified point cloud generation from MuJoCo camera view"""
    # Get camera ID and verify
    cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id == -1:
        raise ValueError(f"Camera '{camera_name}' not found")
    
    # Get camera parameters
    fovy = np.deg2rad(model.cam_fovy[cam_id])
    aspect = renderer.width / renderer.height
    focal = 0.5 * renderer.height / np.tan(0.5 * fovy)
    
    # Get camera pose
    cam_pos = data.cam_xpos[cam_id]
    cam_rot = data.cam_xmat[cam_id].reshape(3, 3)
    
    # Render depth and RGB
    renderer.update_scene(data, camera=camera_name)
    renderer.enable_depth_rendering()
    depth = renderer.render()  # Depth buffer (0-1)
    renderer.disable_depth_rendering()
    rgb = renderer.render()    # RGB image
    
    # Convert depth to metric (using camera clipping planes)
    # near = model.vis.map.znear * model.stat.extent
    # far = model.vis.map.zfar * model.stat.extent
    # depth_metric = far * near / (far - depth * (far - near))
    
    # Create normalized device coordinates
    x = (np.arange(renderer.width) + 0.5 - renderer.width/2) / focal
    y = -(np.arange(renderer.height) + 0.5 - renderer.height/2) / focal  # Flip Y
    x, y = np.meshgrid(x, y)
    
    # Create 3D points in camera space
    points_cam = np.stack((x * depth, 
                          y * depth, 
                          -depth), axis=-1)  # Z points into screen
    # points_cam = points_cam.reshape(-1, 3)
    # Transform to world space
    points_world = (cam_rot @ points_cam.reshape(-1, 3).T).T + cam_pos
    
    # Filter out background points (where depth ≈ 1.0)
    # valid_mask = (depth < 0.999).reshape(-1)
    colors = rgb.reshape(-1, 3)#[valid_mask]
    
    # return points_world[valid_mask], colors
    _save_pointcloud(points=points_world, colors=colors)

def _save_pointcloud(points, colors):
    # DEBUG point - properly add world frame point and color
    points = np.vstack([points, [0, 0, 0]])  # Add origin point
    colors = np.vstack([colors, [255, 0, 0]])  # Color it red (in 0-255 range)

    # Convert to numpy arrays if they aren't already
    points_np = np.asarray(points)
    colors_np = np.asarray(colors) / 255.0  # Convert colors to 0-1 range
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.colors = o3d.utility.Vector3dVector(colors_np)
    # Write the point cloud to a PCD file
    o3d.io.write_point_cloud("point_cloud.pcd", pcd)

def show_pointcloud(pcd_File:str="point_cloud.pcd"):
    cloud = o3d.io.read_point_cloud(pcd_File)
    # Visualize the point cloud
    o3d.visualization.draw_geometries([cloud])
