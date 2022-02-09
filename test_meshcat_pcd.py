import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np
import os
import time

def scale_matrix(factor, origin=None):
    """Return matrix to scale by factor around origin in direction.
    Use factor -1 for point symmetry.
    """
    if not isinstance(factor, list) and not isinstance(factor, np.ndarray):
        M = np.diag([factor, factor, factor, 1.0])
    else:
        assert len(factor) == 3, 'If applying different scaling per dimension, must pass in 3-element list or array'
        #M = np.diag([factor[0], factor[1], factor[2], 1.0])
        M = np.eye(4)
        M[0, 0] = factor[0]
        M[1, 1] = factor[1]
        M[2, 2] = factor[2]
    if origin is not None:
        M[:3, 3] = origin[:3]
        M[:3, 3] *= 1.0 - factor
    return M

def meshcat_pcd_show(mc_vis, point_cloud, color=None, name=None):
    """
    Function to show a point cloud using meshcat. 

    mc_vis (meshcat.Visualizer): Interface to the visualizer 
    point_cloud (np.ndarray): Shape Nx3 or 3xN
    color (np.ndarray or list): Shape (3,)
    """
    if point_cloud.shape[0] != 3:
        point_cloud = np.transpose(point_cloud, axes=(1, 0))
    if color is None:
        color = np.zeros_like(point_cloud) * 255
    if name is None:
        name = 'scene/pcd'

    mc_vis[name].set_object(
        g.Points(
            g.PointsGeometry(point_cloud, color=color),
            g.PointsMaterial()
    ))

def sample_grasp_show(mc_vis, pose_list, idx=None, name=None):
    """
    shows a sample grasp as represented by a bounding box
    """
    if name is None:
        name = 'scene/box'
    if idx is None:
        idx = 0
    sampled_pose = pose_list[idx]
    mc_vis[name].set_object(g.Box([0.1, 0.1, 0.1]))
    mc_vis[name].set_transform(sampled_pose)
    
gt_pcd = np.load('gt_pcd.npy')
obs_pcd = np.load('obs_pcd.npy')
control_label_pcd = np.load('control_pt_1.npy')    
test_pcd = np.load('6d_grasp_building.npy')
control_2 = np.load('control_pt_2.npy')
first_pcd = np.load('first_pcd.npy')

vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
vis['scene'].delete()
print('MeshCat URL: %s' % vis.url())

obs_color = np.zeros_like(obs_pcd)
obs_color[:, 0] = 255*np.ones_like(obs_pcd)[:, 0]
print(obs_color)
meshcat_pcd_show(vis, gt_pcd,  name='scene/gt')
meshcat_pcd_show(vis, obs_pcd, name='scene/obs')
meshcat_pcd_show(vis, control_label_pcd, name='scene/control_pts')
meshcat_pcd_show(vis, test_pcd, name='scene/6d_grasp')
meshcat_pcd_show(vis, control_2, name='scene/control_pts_2')
meshcat_pcd_show(vis, first_pcd, name='scene/first_pcd')
'''
pth = os.path.join('/home/alinasar/acronym/', 'acronym_tools/acronym/data/examples/meshes/Mug/10f6e09036350e92b3f21f1137c3c347.obj')#'models/3b9309c9089549a14ddfc542c04e0efc.obj')
print(pth)
pth2 = os.path.join('/home/alinasar/acronym/acronym_tools/acronym/', 'data/franka_gripper_collision_mesh.stl')
geom = g.ObjMeshGeometry.from_file(pth)
gripper = g.StlMeshGeometry.from_file(pth2)
#print(geom.contents)
vis['scene/object'].set_object(geom)
vis['scene/gripper'].set_object(gripper)
for theta in np.linspace(0, 2 * np.pi, 200):
    vis.set_transform(tf.rotation_matrix(theta, [0, 0, 1]))
    time.sleep(0.005)

#vis['scene/object'].set_transform(scale_matrix(0.001))
'''
