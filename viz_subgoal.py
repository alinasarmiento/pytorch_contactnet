import os
import sys
import numpy as np
import data_utils
import meshcat
from scene_renderer import SceneRenderer
from test_meshcat_pcd import meshcat_pcd_show, sample_grasp_show
import trimesh.transformations as tra
from scipy.spatial.transform import Rotation as R
import model.utils.mesh_utils as mesh_utils
import copy

def create_grasp_arrays(grasp_pose_list, cam_pose_in):
    init_tfs = grasp_pose_list[0]
    goal_tfs = grasp_pose_list[1]
    gripper = mesh_utils.create_gripper('panda', root_folder='./')
    x_r = R.from_euler('y', np.pi)
    x_rot = np.eye(4)
    x_rot[:3, :3] = x_r.as_matrix()
    cam_pose = copy.deepcopy(cam_pose_in)
    cam_pose[:, [1,2]] = -cam_pose[:, [1,2]]
    
    init_gnp = gripper.get_control_point_tensor(init_tfs.shape[0])
    hom_i = np.ones((init_gnp.shape[0], init_gnp.shape[1], 1))
    init_pts = np.concatenate((init_gnp, hom_i), 2).transpose(0,2,1)
    init_pts = np.matmul(init_tfs, init_pts)
    init_pts = np.matmul(np.linalg.inv(cam_pose), init_pts)
    init_pts = init_pts.transpose(0,2,1)
    init_pts = init_pts[:,:,:3]
    
    goal_gnp = gripper.get_control_point_tensor(goal_tfs.shape[0])
    hom_g = np.ones((goal_gnp.shape[0], goal_gnp.shape[1], 1))
    goal_pts = np.concatenate((goal_gnp, hom_g), 2).transpose(0,2,1)
    goal_pts = np.matmul(goal_tfs, goal_pts)
    goal_pts = np.matmul(np.linalg.inv(cam_pose), goal_pts)
    goal_pts = goal_pts.transpose(0,2,1)
    goal_pts = goal_pts[:,:,:3]

    return [init_pts, goal_pts]

def create_scene_arrays():
    data_path = os.fsencode('../acronym/test_scenes/000007.npz')
    data_file = np.load(data_path, fix_imports=True, allow_pickle=True, encoding='bytes')
    obj_paths = data_file['obj_paths']
    scales = data_file['obj_scales']
    init_tfs = data_file['obj_transforms'][0]
    goal_tfs = data_file['obj_transforms'][1]
    success = data_file['scene_contact_points']
    grasp_poses = data_file['grasp_transforms']
    
    paths = []
    for path in obj_paths:
        path = path.split('/')[-1]
        path = '../acronym/models/' + path
        paths.append(path)
        
    pcreader = data_utils.PointCloudReader(data_path)
    sr = SceneRenderer(intrinsics='realsense')

    ##########################
    # camera pose stuff
    
    elevation = 30/180
    az = np.pi
    cam_orientation = tra.euler_matrix(0, -elevation, az)
    coordinate_transform = tra.euler_matrix(np.pi/2, 0, 0).dot(tra.euler_matrix(0, np.pi/2, 0))

    distance = 1.0 #0.9 to 1.3
    extrinsics = np.eye(4)
    extrinsics[0, 3] += distance
    extrinsics = cam_orientation.dot(extrinsics)

    cam_pose = extrinsics.dot(coordinate_transform)
    # table height                                                                                                                                                                                     
    cam_pose[2,3] += sr._table_dims[2]
    cam_pose[:3,:2]= -cam_pose[:3,:2]
    
    ###########################
    grasp_lists = create_grasp_arrays(grasp_poses, cam_pose)
    
    # First, load initial scene
    sr.change_scene(paths, scales, init_tfs)
    rgb, init_depth, init_pc, camera_pose = sr.render(cam_pose) #, render_pc=False)
    
    # Now load goal scene
    sr.change_scene(paths, scales, goal_tfs)
    rgb, goal_depth, goal_pc, camera_pose = sr.render(cam_pose) #, render_pc=False)
    
    np.save('visualization/subgoal_init', init_pc)
    np.save('visualization/subgoal_goal', goal_pc)

    return init_pc, goal_pc, grasp_lists

def visualize_subgoal(init_pc, goal_pc, grasp_lists):

    init_grasps = grasp_lists[0]
    goal_grasps = grasp_lists[1]

    '''
    init_s = success[0]
    goal_s = success[1]

    green_1 = np.zeros_like(init_s)
    green_1[:, 1] = 255*np.ones_like(init_s)[:,1]
    green_2 = np.zeros_like(goal_s)
    green_2[:, 1] = 255*np.ones_like(goal_s)[:,1]

    from IPython import embed; embed()
    '''
    
    vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
    vis['scene'].delete()
    print('MeshCat URL: %s' % vis.url())
    
    meshcat_pcd_show(vis, init_pc, name='scene/sg_init')
    meshcat_pcd_show(vis, goal_pc, name='scene/sg_goal')
    #meshcat_pcd_show(vis, init_s, name='scene/init_s', color=green_1)
    #meshcat_pcd_show(vis, goal_s, name='scene/goal_s', color=green_2)
    sample_grasp_show(vis, init_grasps, name='scene/init/', freq=10)
    sample_grasp_show(vis, goal_grasps, name='scene/goal/', freq=10)
    
if __name__ == '__main__':
    init_pc, goal_pc, grasp_lists = create_scene_arrays()
    visualize_subgoal(init_pc, goal_pc, grasp_lists)
