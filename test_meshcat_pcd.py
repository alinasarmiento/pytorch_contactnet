import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np
import os
import time
import argparse
import trimesh

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

def meshcat_pcd_show(mc_vis, point_cloud, color=None, name=None, size=0.001):
    """
    Function to show a point cloud using meshcat. 
    mc_vis (meshcat.Visualizer): Interface to the visualizer 
    point_cloud (np.ndarray): Shape Nx3 or 3xN
    color (np.ndarray or list): Shape (3,)
    """
    if point_cloud.shape[0] != 3:
        point_cloud = np.transpose(point_cloud, axes=(1, 0))
    if color is None:
        color_pts = np.zeros_like(point_cloud) * 255
    else:
        color_pts = np.zeros_like(point_cloud)
        color_pts[0,:] = color[0]*255
        color_pts[1,:] = color[1]*255
        color_pts[2,:] = color[2]*255
    if name is None:
        name = 'scene/pcd'

    mc_vis[name].set_object(
        g.Points(
            g.PointsGeometry(point_cloud, color=color_pts),
            g.PointsMaterial(size=size)
    ))


def show_mesh(vis, paths, poses, scales, names, clear=False,  opacity=1.0, color=(128,128,128)):
    if vis is None:
        vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
    if clear:
        vis['scene/'].delete()
        vis['home/'].delete()
        vis.delete()
    color = int('%02x%02x%02x' % color, 16)

    for i, (path, pose, scale, name) in enumerate(zip(paths, poses, scales, names)):
        name = name.split('/')[-1]
        trimesh_mesh = trimesh.exchange.load.load(path, file_type='obj')
        if type(trimesh_mesh) == trimesh.Scene:
            meshes = []
            for g in trimesh_mesh.geometry:
                meshes.append(trimesh_mesh.geometry[g])
            trimesh_mesh = trimesh.util.concatenate(meshes)
        scale_tf = trimesh.transformations.scale_matrix(scale)
        trimesh_mesh = trimesh_mesh.apply_transform(scale_tf)
        trimesh_mesh.vertices -= np.mean(trimesh_mesh.vertices, 0)
        
        trimesh_mesh = trimesh_mesh.apply_transform(pose)

        verts = trimesh_mesh.vertices
        faces = trimesh_mesh.faces
        
        material = meshcat.geometry.MeshLambertMaterial(color=color, reflectivity=0.0, opacity=opacity)
        mcg_mesh = meshcat.geometry.TriangularMeshGeometry(verts, faces)
        vis['scene/'+name].set_object(mcg_mesh, material)
    return vis

    
def sample_grasp_show(mc_vis, control_pt_list, name=None, freq=100):
    """
    shows a sample grasp as represented by a little fork guy
    freq: show one grasp per every (freq) grasps (1/freq is ratio of visualized grasps)
    """
    if name is None:
        name = 'scene/loop/'
    for i, gripper in enumerate(control_pt_list):
        color = np.zeros_like(gripper) * 255
        # wrist = gripper[[1, 0, 2], :]
        # wrist = np.transpose(wrist, axes=(1,0))
        
        gripper = gripper[1:,:]
        gripper = gripper[[2, 0, 1, 3], :]
        gripper = np.transpose(gripper, axes=(1,0))
        
        name_i = 'pose'+str(i)
        if i%freq == 0:
            mc_vis[name+name_i].set_object(g.Line(g.PointsGeometry(gripper)))
            #mc_vis[name_i+'wrist'].set_object(g.Line(g.PointsGeometry(wrist)))

def mesh_gripper(mc_vis, pose, name=None, robotiq=False):
    if robotiq:
        gripper_path = os.path.join(os.getenv('HOME'), 'subgoal-net/gripper_models/robotiq_arg2f_base_link.stl')
    else:
        gripper_path = os.path.join(os.getenv('HOME'), 'subgoal-net/gripper_models/panda_gripper/panda_gripper.obj')
    gripper = meshcat.geometry.ObjMeshGeometry.from_file(gripper_path)
    if name is None:
        name = 'gripper'
    mc_vis['scene/'+name].set_object(gripper)
    # print('pose: ', pose)
    #mc_vis['scene/'+name].set_transform(pose)
    mc_vis['scene/'+name].set_transform(pose.astype(np.float64))
            
def viz_pcd(np_pc, name, grasps=False, gripper=False, robotiq=False, clear=False, freq=1):
    vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
    #print('MeshCat URL: %s' % vis.url())
    # vis['scene/'+name].delete()
    if clear:
        vis['scene'].delete()
        vis.delete()
    if grasps:
        sample_grasp_show(vis, np_pc, name=name, freq=freq)
    elif gripper:
        mesh_gripper(vis, np_pc, name=name, robotiq=robotiq)
    else:
        meshcat_pcd_show(vis, np_pc, name=name)


def viz_scene(vis, paths, poses, scales, names, cmeans=None, clear=False, opacity=1.0, goal=False):
    if vis is None:
        vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
    if clear:
        vis['scene/'].delete()
        vis['home/'].delete()
        vis.delete()
    for i, (path, pose, scale, name) in enumerate(zip(paths, poses, scales, names)):
        name = name.split('/')[-1]
        if goal:
            name = 'goal/' + name
        print('path:', path)
        trimesh_mesh = trimesh.exchange.load.load(path, file_type=path.split('.')[-1])
        if type(trimesh_mesh) == trimesh.Scene:
            meshes = []
            for g in trimesh_mesh.geometry:
                meshes.append(trimesh_mesh.geometry[g])
            trimesh_mesh = trimesh.util.concatenate(meshes)

        scale_tf = trimesh.transformations.scale_matrix(scale)
        trimesh_mesh = trimesh_mesh.apply_transform(scale_tf)
        if cmeans is None:
            trimesh_mesh.vertices -= np.mean(trimesh_mesh.vertices,0) #trimesh_mesh.center_mass
        else:
            trimesh_mesh.vertices -= cmeans[i]
        trimesh_mesh = trimesh_mesh.apply_transform(pose)

        verts = trimesh_mesh.vertices
        faces = trimesh_mesh.faces

        color=(128, 128, 128)
        color = int('%02x%02x%02x' % color, 16)

        material = meshcat.geometry.MeshLambertMaterial(color=color, reflectivity=0.0, opacity=opacity)
        mcg_mesh = meshcat.geometry.TriangularMeshGeometry(verts, faces)
        vis['scene/'+name].set_object(mcg_mesh, material)
        # print('pose: ', pose)
    return vis
        
            
def visualize(args):
    vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
    vis['scene'].delete()
    vis.delete()
    print('MeshCat URL: %s' % vis.url())

    pb_pcd = np.load('pybullet_pcd.npy')
    
    cam_pose = np.load('cam_pose.npy')
    box = meshcat.geometry.Box([0.1, 0.2, 0.3])
    vis['scene/cam'].set_object(box)
    vis['scene/cam'].set_transform(cam_pose)
    
    threshold = 0.2
    pred_s = np.load('pred_s_mask.npy')
    print(np.max(pred_s))
    success_mask = np.where(pred_s > threshold)
    pcd = np.load('full_pcd.npy')
    s_pcd = pcd[success_mask]
    print(success_mask)
    green = np.zeros_like(s_pcd)
    green[:, 1] = 255*np.ones_like(s_pcd)[:,1]

    pc_world = np.load('world_pc.npy')
    pc_cam = np.load('cam_pc.npy')
    if args.i is not None:
        grasps = np.load('control_pt_list.npy')[:args.i]
        grasp_labels = np.load('label_pt_list.npy')[:args.i]
    else:
        grasps = np.load('control_pt_list.npy')
        grasp_labels = np.load('label_pt_list.npy')
        
    print('pred', grasps.shape)
    pc_gt = np.load('ground_truth.npy')
    print('labels', grasp_labels.shape)

    pose = np.eye(4)
    pose[:3, :3] = pcd[0]

    #meshcat_pcd_show(vis, pb_pcd, name='scene/pb')

    meshcat_pcd_show(vis, pc_world, name='scene/world')
    print('show world pc')
    meshcat_pcd_show(vis, pc_gt, name='scene/gt')
    meshcat_pcd_show(vis, s_pcd, name='scene/s_pcd', color=green.T)
    sample_grasp_show(vis, grasps, name='pred/', freq=1)

    d = np.load('d.npy')
    meshcat_pcd_show(vis, d, name='scene/d')

    
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
    parser.add_argument('--i', type=int, default=None, help='index for single grasp viz')
    args = parser.parse_args()
    
    visualize(args)
