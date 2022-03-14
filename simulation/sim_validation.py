import os.path
import pybullet as p
import torch
import sys
from trac_ik_python import trac_ik
from scipy.spatial.transform import Rotation as R
from airobot import Robot
from airobot.franka_pybullet import FrankaPybullet
from panda_pb_cfg import get_cfg_defaults
from airobot.sensor.camera.rgbdcam_pybullet import RGBDCameraPybullet

sys.path.append('../')
import numpy as np
from model.contactnet import ContactNet
import model.utils.config_utils as config_utils
from dataset import ContactDataset
from train import initialize_loaders, initialize_net
from scipy.spatial.transform import Rotation as R
import argparse
from torch_geometric.nn import fps

def _camera_cfgs():
    """                                                                                                                                                                                                
    Returns a set of camera config parameters                                                                                                                                                          

    Returns:                                                                                                                                                                                           
        YACS CfgNode: Cam config params                                                                                                                                                                
    """
    _C = CN()
    _C.ZNEAR = 0.01
    _C.ZFAR = 10
    _C.WIDTH = 640
    _C.HEIGHT = 480
    _C.FOV = 60
    _ROOT_C = CN()
    _ROOT_C.CAM = CN()
    _ROOT_C.CAM.SIM = _C
    return _ROOT_C.clone()

def get_rand_scene(dataset):
    scene_idx = math.randint(0, len(dataset.data))
    data_file = dataset.data[scene_idx]
    filename = '../acronym/scene_contacts/' + os.fsdecode(data_file)
    scene_data = load(filename)

    obj_paths = scene_data['obj_paths']
    for i, path in enumerate(obj_paths):
        fixed_path = '../acronym/models/' + path.split('/')[-1]
        obj_paths[i] = fixed_path
    obj_scales = scene_data['obj_scales']
    obj_transforms = scene_data['obj_transforms']
    return obj_scales, obj_transforms, obj_paths
    
def infer(model, dataset, config):
    model.eval()

    # Load the panda and the scene into pybullet
    panda_cfg = get_cfg_defaults()
    panda_ar = Robot('panda',
                      pb=True,
                      pb_cfg={'gui': False,
                              'opengl_render': False},
                      arm_cfg={'self_collision': False,
                               'seed': None}))
    #pb_client = p.connect(p.DIRECT)

    # Get a random cluttered scene from the dataset
    obj_scales, obj_transforms, obj_paths = get_rand_scene(dataset)
    
    p.setGravity(0, 0, -9.8)

    cam_cfg = {}
    cam_cfg['focus_pt'] = panda_cfg.CAMERA_FOCUS
    cam_cfg['dist'] = [0.8, 0.8, 0.8, 0.8]
    cam_cfg['yaw'] = [30, 150, 210, 330]
    cam_cfg['pitch'] = [-35, -35, -20, -20]
    cam_cfg['roll'] = [0, 0, 0, 0]

    camera = RGBDCameraPybullet(cfgs=_camera_cfgs(), pb_client=panda_ar.pb_client)
    camera.setup_camera(
        focus_pt=cam_cfg['focus_pt'],
        dist=cam_cfg['dist'],
        yaw=cam_cfg['yaw'],
        pitch=cam_cfg['pitch'],
        roll=cam_cfg['roll'])
    
    collision_args = {}
    visual_args = {}
    pb_ids = []
    for path, scale, transform in zip(obj_paths, obj_scales, obj_transforms):
        rot_mat = np.array(transform[:3, :3])
        t = np.array(transform[:, 3]).T
        q = R.from_matrix(rot_mat).as_quat()
        
        collision_args['collisionFramePosition'] = t
        collision_args['collisionFrameOrientation'] = q
        visual_args['visualFramePosition'] = t
        visual_args['visualFrameOrientation'] = q
        
        collision_args['shapeType'] = p.GEOM_MESH
        collision_args['fileName'] = path
        collision_args['meshScale'] = scale
        visual_args['shapeType'] = p.GEOM_MESH
        visual_args['fileName'] = path
        visual_args['meshScale'] = scale
        visual_args['rgbaColor'] = rgba
	visual_args['specularColor'] = specular

        vs_id = p.createVisualShape(**visual_args)
        cs_id = p.createCollisionShape(**collision_args)
        body_id = p.createMultiBody(baseMass=1.0,
                                       baseInertialFramePosition=t,
                                       baseInertialFrameOrientation=q,
                                       baseCollisionShapeIndex=cs_id,
                                       baseVisualShapeIndex=vs_id,
                                       basePosition=t,
                                       baseOrientation=q,
                                       **kwargs)
        pb_ids.append(body_id)
        
    # Render pointcloud...?
    rgb, depth, seg = cam.get_images(
                get_rgb=True,
                get_depth=True,
                get_seg=True)
    pcd, colors_raw = cam.get_pcd(
                    in_world=True,
                    filter_depth=False,
                    depth_max=1.0)

    # Forward pass into model
    batch = torch.ones(length(pcd), 1)
    idx = fps(pcd, torch.ones(batch, 2048/length(pcd)))
    points, pred_grasps, pred_successes, pred_widths = model(pcd[:, 3:], pos=pcd[:, :3], batch=batch, idx=idx, k=None)

    # Return a SINGLE predicted grasp
    top_grasp = pred_grasps[argmax(pred_successes)]
    grasp_width = pred_widths[argmax(pred_successes)]
    pb_execute(franka_ar, top_grasp, grasp_width)
              
def pb_execute(robot, grasp_pose, grasp_width):
    rot = R.from_matrix(grasp_pose[:3, :3])
    quat = rot.as_quat()
    pos = grasp_pose[:3, -1]
    num_ik_solver = trac_ik.IK('panda_link0', 'panda_hand', urdf_string='franka.urdf')
    jpos = robot.arm.get_jpos()
    sol_jnts = num_ik_solver.get_ik(seed=jpos,
                                    pos[0],
                                    pos[1],
                                    pos[2],
                                    quat[0],
                                    quat[1],
                                    quat[2],
                                    quat[3])
    success = False
    robot.arm.eetool.open()
    robot.arm.eetool.set_jpos(grasp_width, wait=True, ignore_physics=False)
    robot.arm.set_jpos(sol_jnts, wait=False)
    return success
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, help='path to load model from')
    parser.add_argument('--config_path', type=str, default='./model/', help='path to config yaml file')
    parser.add_argument('--data_path', type=str, default='/home/alinasar/acronym/scene_contacts', help='path to acronym dataset with Contact-GraspNet folder')
    args = parser.parse_args()

    contactnet, config = initialize_net(args.config_path, load_model=True, save_path=args.save_path)
    dataset = ContactDataset(args.data_path, config)
    robot, predicted_grasp = infer(contactnet, dataset, config)
    successful = pb_execute(robot, predicted_grasp)
