import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np
import os
import time
import argparse

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

def sample_grasp_show(mc_vis, control_pt_list, name=None, freq=100):
    """
    shows a sample grasp as represented by a little fork guy
    freq: show one grasp per every (freq) grasps (1/freq is ratio of visualized grasps)
    """
    if name is None:
        name = 'scene/loop/'
    for i, gripper in enumerate(control_pt_list):
        color = np.zeros_like(gripper) * 255
        wrist = gripper[[1, 0, 2], :]
        wrist = np.transpose(wrist, axes=(1,0))
        
        gripper = gripper[1:,:]
        gripper = gripper[[2, 0, 1, 3], :]
        gripper = np.transpose(gripper, axes=(1,0))
        
        name_i = 'pose'+str(i)
        if i%freq == 0:
            mc_vis[name+name_i].set_object(g.Line(g.PointsGeometry(gripper)))
            #mc_vis[name_i+'wrist'].set_object(g.Line(g.PointsGeometry(wrist)))

def visualize(args):
    vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
    vis['scene'].delete()
    vis.delete()
    print('MeshCat URL: %s' % vis.url())

    cam_pose = np.load('cam_pose.npy')
    box = meshcat.geometry.Box([0.1, 0.2, 0.3])
    vis['scene/cam'].set_object(box)
    vis['scene/cam'].set_transform(cam_pose)
    
    threshold = 0.2
    pred_s = np.load('pred_s_mask.npy')
    print(np.max(pred_s))
    success_mask = np.where(pred_s > threshold)
    pcd = np.load('full_pcd.npy')
    print(pred_s.shape, pcd.shape)
    s_pcd = pcd[success_mask]
    print(success_mask)
    green = np.zeros_like(s_pcd)
    green[:, 1] = 255*np.ones_like(s_pcd)[:,1]

    pc_world = np.load('world_pc.npy')
    pc_cam = np.load('cam_pc.npy')
    grasps = np.load('control_pt_list.npy')
    grasp_labels = np.load('label_pt_list.npy')
    
    meshcat_pcd_show(vis, pcd, name='scene/full_pcd')
    meshcat_pcd_show(vis, s_pcd, name='scene/s_pcd', color=green.T)
    meshcat_pcd_show(vis, pc_world, name='scene/world')
    meshcat_pcd_show(vis, pc_cam, name='scene/pc_cam')
    sample_grasp_show(vis, grasps, name='pred/', freq=1)
    sample_grasp_show(vis, grasp_labels, name='label/', freq=1)
    '''
    obs_color = np.zeros_like(pos_pcd)
    obs_color[:, 0] = 255*np.ones_like(pos_pcd)[:, 0]
    green_color = np.zeros_like(pos_labeled)
    green_color[:, 1] = 255*np.ones_like(pos_labeled)[:, 1]
    white_color = 255*np.ones_like(pred_s_pcd)
    '''
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--single_grasp', type=bool, default=False, help='load a single grasp with contact point emphasized')
    args = parser.parse_args()
    
    visualize(args)
