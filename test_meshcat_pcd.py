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
    shows a sample grasp as represented by a bounding box
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
    #control_expanded = np.load('control_pt_list_many.npy')
    #label_expanded = np.load('label_pt_list_many.npy')
    #s_grasp_expanded = np.load('success_pt_list_many.npy')
    pybullet = np.load('pybullet_pcd.npy')
    p = np.load('first_pcd_many.npy')
    vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
    vis['scene'].delete()
    print('MeshCat URL: %s' % vis.url())

    '''
    obs_color = np.zeros_like(pos_pcd)
    obs_color[:, 0] = 255*np.ones_like(pos_pcd)[:, 0]
    green_color = np.zeros_like(pos_labeled)
    green_color[:, 1] = 255*np.ones_like(pos_labeled)[:, 1]
    white_color = 255*np.ones_like(pred_s_pcd)
    '''
    #sample_grasp_show(vis, control_expanded, name='scene/control_loops/')
    #sample_grasp_show(vis, label_expanded, name='scene/label_gripper/')
    #sample_grasp_show(vis, s_grasp_expanded, name='scene/success_masked/', freq=100)

    meshcat_pcd_show(vis, pybullet, name='scene/pb')
    #meshcat_pcd_show(vis, p, name='scene/pcd')
    #meshcat_pcd_show(vis, gt_pcd,  name='scene/gt')
    #meshcat_pcd_show(vis, obs_pcd, name='scene/obs')
    #meshcat_pcd_show(vis, control_label_pcd, name='scene/control_pts')
    #meshcat_pcd_show(vis, control_2, name='scene/control_pts_2')
    #meshcat_pcd_show(vis, first_pcd, name='scene/first_pcd')
    #meshcat_pcd_show(vis, pred_s_pcd, name='scene/pred_s') #, color=white_color)
    #meshcat_pcd_show(vis, pos_pcd, name='scene/pos_pcd', color=obs_color.T)
    #meshcat_pcd_show(vis, pos_labeled, name='scene/pos_labeled', color=green_color.T)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--single_grasp', type=bool, default=False, help='load a single grasp with contact point emphasized')
    args = parser.parse_args()
    
    visualize(args)
