import os.path
import pybullet as p
import torch
import sys
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

sys.path.append('pybullet-planning/')
from pybullet_tools.utils import add_data_path, connect, dump_body, disconnect, wait_for_user, \
    get_movable_joints, get_sample_fn, set_joint_positions, get_joint_name, LockRenderer, link_from_name, get_link_pose, \
    multiply, Pose, Point, interpolate_poses, HideOutput, draw_pose, set_camera_pose, load_pybullet, \
    assign_link_colors, add_line, point_from_pose, remove_handles, BLUE, pairwise_collision, set_client, get_client, pairwise_link_collision, \
    plan_joint_motion
from pybullet_tools.ikfast.franka_panda.ik import PANDA_INFO, FRANKA_URDF
from pybullet_tools.ikfast.ikfast import get_ik_joints, either_inverse_kinematics, check_ik_solver

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
    '''
    Handles pybullet scene generation and grasp inference.
        1. creates a mesh scene in pybullet from a random scene in the dataset
        2. renders a pointcloud from pybullet from the scene
        3. infers a grasp using a loaded pretrained model
    '''
    model.eval()

    # Load the panda and the scene into pybullet
    panda_cfg = get_cfg_defaults()
    panda_ar = Robot('panda',
                      pb=True,
                      pb_cfg={'gui': False,
                              'opengl_render': False},
                      arm_cfg={'self_collision': False,
                               'seed': None})

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
    rgb, depth, seg = camera.get_images(
                get_rgb=True,
                get_depth=True,
                get_seg=True)
    pcd, colors_raw = camera.get_pcd(
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
    return franka_ar, top_grasp, grasp_width

def check_collision(pb_robot):
    pb_client = get_client()
    check_self_coll_pairs = []
    for i in range(p.getNumJoints(pb_robot, physicsClientId=pb_client)):
        for j in range(p.getNumJoints(pb_robot, physicsClientId=pb_client)):
            # don't check link colliding with itself, and ignore specified links
            if i != j and (i, j) not in ignore_link_pairs:
                check_self_coll_pairs.append((i, j))
    for link1, link2 in self.check_self_coll_pairs:
         if pairwise_link_collision(self.robot, link1, self.robot, link2):
             return True, 'self'
    #TO-DO: add checking between robot and objects in scene
    # use body_collision(pb_robot, [object]) and parse thru pybullet bodies

def solve_ik(pose):
    tool_link = link_from_name(pb_robot, 'panda_hand')
    ik_joints = get_ik_joints(pb_robot, PANDA_INFO, tool_link)
    confs = either_inverse_kinematics(pb_robot, PANDA_INFO, tool_link, pose,
                                      max_distance=None, max_time=0.5, max_candidates=250)
    for conf in confs:
        set_joint_positions(pb_robot, ik_joints, conf)
        collision_info = check_collision(pb_robot)
        if not collision_info[0]:
            return conf 
        else:
            print('Collision with body: %s' % collision_info[1])
    print('Failed to get feasible IK')
    return None
    
def pb_execute(robot, pb_robot, grasp_pose, grasp_width):
    '''
    Finds IK solution for given grasp pose and robot (using pybullet-planner)
    args
        robot: airobot instance
        pb_robot: pybullet robot as instantiated by load_pybullet
        grasp_pose: 4x4 grasp pose matrix (output of contactnet model)
        grasp_width: scalar
    returns
        success (bool)
    '''
    rot = R.from_matrix(grasp_pose[:3, :3])
    quat = rot.as_quat()
    pos = grasp_pose[:3, -1]
    lift_pos = copy.deepcopy(pos)
    lift_pos[2] += 0.3

    # ik stuff
    pose = (tuple(pos), tuple(quat))
    lift_pose = (tuple(lift_pos), tuple(quat))
    sol_jnts = solve_ik(pose)
    lift_jnts = solve_ik(lift_pose)

    success = False # this needs to be set
    robot.arm.eetool.open()
    robot.arm.set_jpos(sol_jnts, wait=False, ignore_physics=True)
    robot.arm.eetool.set_jpos(grasp_width-0.01, wait=True, ignore_physics=True)
    robot.arm.set_jpos(lift_jnts, wait=False)
    # wait
    # check contact between fingers and object (?)
    return success
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='./checkpoints/model_save.pth', help='path to load model from')
    parser.add_argument('--config_path', type=str, default='./model/', help='path to config yaml file')
    parser.add_argument('--data_path', type=str, default='/home/alinasar/acronym/scene_contacts', help='path to acronym dataset with Contact-GraspNet folder')
    args = parser.parse_args()

    with LockRenderer():
        with HideOutput(True):
            pb_robot = load_pybullet(FRANKA_URDF, base_pos=[0, 0, 1], fixed_base=True)
            assign_link_colors(pb_robot, max_colors=3, s=0.5, v=1.)
    
    contactnet, config = initialize_net(args.config_path, load_model=True, save_path=args.save_path)
    dataset = ContactDataset(args.data_path, config)
    robot, predicted_grasp, predicted_width = infer(contactnet, dataset, config)
    successful = pb_execute(robot, pb_robot, predicted_grasp, predicted_width)
