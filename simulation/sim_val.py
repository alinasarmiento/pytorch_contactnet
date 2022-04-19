import os
import os.path
import pybullet as p
import torch
import sys
from scipy.spatial.transform import Rotation as R
import math
from yacs.config import CfgNode as CN
import copy
import trimesh
import random

sys.path[0] += '/../'
print(os.getcwd())
print(sys.path)
import numpy as np
from model.contactnet import ContactNet
import model.utils.config_utils as config_utils
from dataset import ContactDataset
from train import initialize_loaders, initialize_net
from scipy.spatial.transform import Rotation as R
import argparse
from torch_geometric.nn import fps

sys.path.append('../')
sys.path.append('../airobot/src/')
from airobot import Robot
#from airobot.franka_pybullet import FrankaPybullet
from airobot.utils.pb_util import BulletClient
from airobot.sensor.camera.rgbdcam_pybullet import RGBDCameraPybullet
sys.path.remove('../airobot/src/')
sys.path[0] = sys.path[0].rstrip(sys.path[0][-4:])
from panda_pb_cfg import get_cfg_defaults
#from franka_ik import FrankaIK

sys.path[0] += '/../../pybullet-planning/'
from pybullet_tools.utils import add_data_path, connect, dump_body, disconnect, wait_for_user, \
    get_movable_joints, get_sample_fn, set_joint_positions, get_joint_name, LockRenderer, link_from_name, get_link_pose, \
    multiply, Pose, Point, interpolate_poses, HideOutput, draw_pose, set_camera_pose, load_pybullet, \
    assign_link_colors, add_line, point_from_pose, remove_handles, BLUE, pairwise_collision, set_client, get_client, pairwise_link_collision, \
    plan_joint_motion
from pybullet_tools.ikfast.franka_panda.ik import PANDA_INFO, FRANKA_URDF
from pybullet_tools.ikfast.ikfast import get_ik_joints, either_inverse_kinematics, check_ik_solver
FRANKA_URDF = os.path.join(sys.path[0], FRANKA_URDF)

class PandaPB():
    def __init__(self, contactnet, dataset, config, gui=True):
        
        self.gui = gui
        if self.gui:
            set_client(0) #self.robot.pb_client)
        else:
            set_client(1)
        connect(use_gui=gui)
        with LockRenderer():
            with HideOutput(False):
                 self.pb_robot = load_pybullet(FRANKA_URDF, base_pos=[-0.3, 0.3, 0.3], fixed_base=True)
                 assign_link_colors(self.pb_robot, max_colors=3, s=0.5, v=1.)
        self.model = contactnet
        self.dataset = dataset
        self.config = config
        self.pb_ids = None
        self.robot = None
        
    def _camera_cfgs(self):
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

    def get_rand_scene(self):
        scene_idx = 0 #np.random.randint(0, len(self.dataset.data))
        data_file = self.dataset.data[scene_idx]
        filename = '../acronym/scene_contacts/' + os.fsdecode(data_file)
        scene_data = np.load(filename, allow_pickle=True)
        obj_paths = scene_data['obj_paths']
        for i, path in enumerate(obj_paths):
            fixed_path = '../acronym/models/' + path.split('/')[-1]
            obj_paths[i] = fixed_path
        obj_scales = scene_data['obj_scales']
        obj_transforms = scene_data['obj_transforms']
        return obj_scales, obj_transforms, obj_paths

    def infer(self):
        '''
        Handles pybullet scene generation and grasp inference.
            1. creates a mesh scene in pybullet from a random scene in the dataset
            2. renders a pointcloud from pybullet from the scene
            3. infers a grasp using a loaded pretrained model
        '''
        self.model.eval()

        # Load the panda and the scene into pybullet
        panda_cfg = get_cfg_defaults()
        self.robot = Robot('franka',
                          pb=True,
                          pb_cfg={'gui': False,
                                  'opengl_render': False},
                          arm_cfg={'self_collision': False,
                                   'seed': None})
        
        self.robot.pb_client.set_step_sim(True)
        # Get a random cluttered scene from the dataset
        obj_scales, obj_transforms, obj_paths = self.get_rand_scene()

        p.setGravity(0, 0, -9.8)

        cam_cfg = {}
        cam_cfg['focus_pt'] = panda_cfg.CAMERA_FOCUS
        cam_cfg['dist'] = [0.8, 0.8, 0.8, 0.8]
        cam_cfg['yaw'] = [30, 150, 210, 330]
        cam_cfg['pitch'] = [-35, -35, -20, -20]
        cam_cfg['roll'] = [0, 0, 0, 0]

        self.camera = RGBDCameraPybullet(cfgs=self._camera_cfgs(), pb_client=self.robot.pb_client)
        self.camera.setup_camera(
            focus_pt=[0, 0, 0], #cam_cfg['focus_pt'])
            dist=1, yaw=0, pitch=-40)#cam_cfg['dist'],
            #yaw=cam_cfg['yaw'],
            #pitch=cam_cfg['pitch'],
            #roll=cam_cfg['roll'])

        collision_args = {}
        visual_args = {}
        self.pb_ids = []
        i = 0
        for path, scale, transform in zip(obj_paths, obj_scales, obj_transforms):
            name = 'object_' + str(i) + '.obj'
            i += 1
            print(transform)
            rot_mat= np.array(transform[:3, :3])
            t = np.array(transform[:, 3]).T[:3]
            q = R.from_matrix(rot_mat).as_quat()

            tmesh = trimesh.load(path, process=False)
            if isinstance(tmesh, trimesh.Scene):
                tmesh_merge = trimesh.util.concatenate(
                        tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                            for g in tmesh.geometry.values()))
            else:
                tmesh_merge = tmesh
            tmp_path = os.path.join(os.getcwd(), name)
            tmesh_merge.vertices -= tmesh.centroid
            tmesh_merge.export(tmp_path)

            print(t, q)
            collision_args['collisionFramePosition'] = None
            collision_args['collisionFrameOrientation'] = None
            visual_args['visualFramePosition'] = None
            visual_args['visualFrameOrientation'] = None

            collision_args['shapeType'] = p.GEOM_MESH
            collision_args['fileName'] = tmp_path
            collision_args['meshScale'] = np.array([1,1,1])*scale
            visual_args['shapeType'] = p.GEOM_MESH
            visual_args['fileName'] = tmp_path
            visual_args['meshScale'] = np.array([1,1,1])*scale
            visual_args['rgbaColor'] = [0.5, 0, 0, 1] #rgba
            visual_args['specularColor'] = [0, 0.5, 0.4]#specular

            print(path)
            print(scale)
            vs_id = p.createVisualShape(**visual_args)
            cs_id = p.createCollisionShape(**collision_args)
            body_id = p.createMultiBody(baseMass=1.0,
                                           baseInertialFramePosition=None, #t,
                                           baseInertialFrameOrientation=None, #q,
                                           baseCollisionShapeIndex=cs_id,
                                           baseVisualShapeIndex=vs_id,
                                           basePosition=t,
                                           baseOrientation=q)
                                           #**kwargs)
            self.pb_ids.append(body_id)
            t[2] -= 0.3 #bring the objects down onto the floor for airobot rendering
            self.robot.pb_client.load_geom(shape_type='mesh', visualfile=tmp_path, collifile=tmp_path,
                                     mass=1.0, mesh_scale=scale, rgba=[0.5,0,0,1], specular=[0,0.5,0.4],
                                     base_pos=t, base_ori=q)
            
        # Add table
        '''
        collision_args = {}
        visual_args = {}
        t = np.array([0,0,0.295])
        q = np.array([0,0,0,1])
        collision_args['collisionFramePosition'] = None
        collision_args['collisionFrameOrientation'] = None
        visual_args['visualFramePosition'] = None
        visual_args['visualFrameOrientation'] = None

        collision_args['shapeType'] = p.GEOM_BOX
        collision_args['halfExtents'] = np.array([1,1,0.005])
        visual_args['shapeType'] = p.GEOM_BOX
        visual_args['halfExtents'] = np.array([1,1,0.005])
        visual_args['rgbaColor'] = [0, 0, 0.5, 1]
        visual_args['specularColor'] = [0, 0.5, 0.4]
        vs_id = p.createVisualShape(**visual_args)
        cs_id = p.createCollisionShape(**collision_args)
        body_id = p.createMultiBody(baseMass=1.0,
                                       baseInertialFramePosition=None, #t,                                                                                                            
                                       baseInertialFrameOrientation=None, #q,                                                                                                         
                                       baseCollisionShapeIndex=cs_id,
                                       baseVisualShapeIndex=vs_id,
                                       basePosition=t,
                                       baseOrientation=q)
                                       #**kwargs)                                                                                                                                     
        self.pb_ids.append(body_id)
        '''    
        # Render pointcloud...
        
        rgb, depth, seg = self.camera.get_images(
                    get_rgb=True,
                    get_depth=True,
                    get_seg=True)
        pcd, colors_raw = self.camera.get_pcd(
            in_world=True,
            filter_depth=True,
            depth_min=-5.0,
            depth_max=5.0)

        # crop pointcloud and potentially center it as well
        x_mask = np.where((pcd[:, 0] < 0.75) & (pcd[:, 0] > -0.75))
        y_mask = np.where((pcd[:, 1] < 0.75) & (pcd[:, 1] > -0.75))
        z_mask = np.where((pcd[:, 2] < 0.25)) # & (pcd[:, 2] > 0.04))
        mask = np.intersect1d(x_mask, y_mask)
        mask = np.intersect1d(mask, z_mask)
        pcd = pcd[mask]
        
        # go from world frame to camera frame
        world2cam = np.linalg.inv([self.camera.cam_ext_mat])
        pcd = np.concatenate((pcd, np.ones((pcd.shape[0],1))), axis=1)
        pcd = np.transpose(pcd, (1,0))
        world2cam = np.tile(world2cam, (pcd.shape[0], 1, 1))
        pcd = np.matmul(world2cam, pcd)
        pcd = np.transpose(pcd, (2, 0, 1))
        pcd  = pcd[:, 0, :3]

        # Forward pass into model
        downsample = np.array(random.sample(range(pcd.shape[0]-1), 20000)) #np.linspace(0, pcd.shape[0]-1, 20000, dtype=int)
        pcd = pcd[downsample, :]
        np.save('pybullet_pcd.npy', pcd)        
        print('here')
        '''
        import matplotlib.pyplot as plt
        plt.imshow(rgb)
        plt.show()
        plt.imshow(depth)
        plt.show()
        tpcd = trimesh.PointCloud(pcd)
        # tpcd.show()
        '''
        
        pcd = torch.Tensor(pcd).to(dtype=torch.float32).to(self.model.device) 
        batch = torch.ones(pcd.shape[0]).to(dtype=torch.int64).to(self.model.device)
        idx = torch.linspace(0, pcd.shape[0]-1, 2048).to(dtype=torch.int64).to(self.model.device) #fps(pcd, batch, 2048/pcd.shape[0])
        points, pred_grasps, pred_successes, pred_widths = self.model(pcd[:, 3:], pos=pcd[:, :3], batch=batch, idx=idx, k=None)
        print('model pass')
        pred_grasps = torch.flatten(pred_grasps, start_dim=0, end_dim=1)
        pred_successes = torch.flatten(pred_successes)
        pred_widths = torch.flatten(pred_widths, start_dim=0, end_dim=1)
        points = torch.flatten(points, start_dim=0, end_dim=1)

        # Return a SINGLE predicted grasp
        point = points[torch.argmax(pred_successes)]
        top_grasp = pred_grasps[torch.argmax(pred_successes)]
        grasp_width = pred_widths[torch.argmax(pred_successes)]
        print(point.shape)
        print(top_grasp.shape)
        
        # transform back into the world frame
        cam2world = np.array([self.camera.cam_ext_mat])
        top_grasp = np.matmul(cam2world, top_grasp.detach().cpu())[0]
        point = np.matmul(cam2world, np.concatenate((point.detach().cpu(), [1])))[0, :3]

        from IPython import embed; embed()
        
        return top_grasp, grasp_width.detach().cpu(), point, torch.max(pred_successes).detach().cpu()

    def check_collision(self):
        pb_client = get_client()
        check_self_coll_pairs = []
        for i in range(p.getNumJoints(self.pb_robot, physicsClientId=pb_client)):
            for j in range(p.getNumJoints(self.pb_robot, physicsClientId=pb_client)):
                # don't check link colliding with itself, and ignore specified links
                if i != j and (i, j) not in ignore_link_pairs:
                    check_self_coll_pairs.append((i, j))
        for link1, link2 in self.check_self_coll_pairs:
             if pairwise_link_collision(self.robot, link1, self.robot, link2):
                 return True, 'self'

        # check collisions between robot and pybullet scene bodies
        for body in self.pb_ids:
            collision = body_collision(self.pb_robot, body)
            if collision:
                return True, body
        return False

    def solve_ik(self, pose):
        tool_link = link_from_name(self.pb_robot, 'panda_hand')
        ik_joints = get_ik_joints(self.pb_robot, PANDA_INFO, tool_link)
        print(ik_joints)
        confs = either_inverse_kinematics(self.pb_robot, PANDA_INFO, tool_link, pose,
                                          max_distance=None, max_time=0.5, max_candidates=250)
        for conf in confs:
            print(conf)
            set_joint_positions(self.pb_robot, ik_joints, conf)
            collision_info = self.check_collision(self.pb_robot)
            if not collision_info[0]:
                return conf 
            else:
                print('Collision with body: %s' % collision_info[1])
        print('Failed to get feasible IK')
        return None

    def pb_execute(self):
        '''
        Finds IK solution for given grasp pose and robot (using pybullet-planner)
        args
            robot: airobot instance
            grasp_pose: 4x4 grasp pose matrix (output of contactnet model)
            grasp_width: scalar
        returns
            success (bool)
        '''
        predicted_grasp, predicted_width, point, s = self.infer()


        ############
        # VISUALIZATION IN OPENGL

        rot = R.from_matrix(predicted_grasp[:3, :3])

        quat = rot.as_quat()
        pos = predicted_grasp[:3, -1]
        pos[2] += 0.3
        lift_pos = copy.deepcopy(pos)
        lift_pos[2] += 0.3

        t = pos
        q = quat
        collision_args = {}
        visual_args = {}
        panda_path = './gripper_models/panda_gripper/panda_gripper.obj'

        print(t, q)
        collision_args['collisionFramePosition'] = None
        collision_args['collisionFrameOrientation'] = None
        visual_args['visualFramePosition'] = None
        visual_args['visualFrameOrientation'] = None

        collision_args['shapeType'] = p.GEOM_MESH
        # collision_args['fileName'] = path                                                                                                                                           
        collision_args['fileName'] = panda_path
        collision_args['meshScale'] = np.array([1,1,1])
        visual_args['shapeType'] = p.GEOM_MESH
        # visual_args['fileName'] = path                                                                                                                                              
        visual_args['fileName'] = panda_path
        visual_args['meshScale'] = np.array([1,1,1])
        visual_args['rgbaColor'] = [0, 0, 0, 0.5] #rgba                                                                                                                               
        visual_args['specularColor'] = [0, 0.5, 0.4]#specular                                                                                                                         

        vs_id = p.createVisualShape(**visual_args)
        cs_id = p.createCollisionShape(**collision_args)
        body_id = p.createMultiBody(baseMass=1.0,
                                       baseInertialFramePosition=None,
                                       baseInertialFrameOrientation=None,             
                                       baseCollisionShapeIndex=cs_id,
                                       baseVisualShapeIndex=vs_id,
                                       basePosition=t,
                                       baseOrientation=q)
                                       #**kwargs)

        q = [0,0,0,1]
        t = point
        t[2] += 0.3
        proj = self.camera.proj_matrix
        
        collision_args = {}
        visual_args = {}

        collision_args['collisionFramePosition'] = None
        collision_args['collisionFrameOrientation'] = None
        visual_args['visualFramePosition'] = None
        visual_args['visualFrameOrientation'] = None

        collision_args['shapeType'] = p.GEOM_SPHERE
        collision_args['radius'] = 0.01
        visual_args['shapeType'] = p.GEOM_SPHERE
        visual_args['radius'] = 0.01

        visual_args['rgbaColor'] = [0, 0, 0, 0.5] #rgba                                  
        visual_args['specularColor'] = [0, 0.5, 0.4]#specular

        vs_id = p.createVisualShape(**visual_args)
        cs_id = p.createCollisionShape(**collision_args)
        body_id = p.createMultiBody(baseMass=1.0,
                                       baseInertialFramePosition=None,                               
                                       baseInertialFrameOrientation=None,
                                       baseCollisionShapeIndex=cs_id,
                                       baseVisualShapeIndex=vs_id,
                                       basePosition=t,
                                       baseOrientation=q)
                                       #**kwargs)

                                       
        from IPython import embed; embed()
        #####################
        
        # ik stuff
        pose = (tuple(pos), tuple(quat))
        lift_pose = (tuple(lift_pos), tuple(quat))
        sol_jnts = self.solve_ik(pose)
        lift_jnts = self.solve_ik(lift_pose)

        success = False # this needs to be set
        self.robot.arm.eetool.open()
        self.robot.arm.set_jpos(sol_jnts, wait=False, ignore_physics=True)
        self.robot.arm.eetool.set_jpos(predicted_width-0.01, wait=True, ignore_physics=True)
        self.robot.arm.set_jpos(lift_jnts, wait=False)
        # wait
        # check contact between fingers and object (?)
        return success
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='./checkpoints/march_save2.pth', help='path to load model from')
    parser.add_argument('--config_path', type=str, default='./eval/', help='path to config yaml file')
    parser.add_argument('--data_path', type=str, default='/home/alinasar/acronym/scene_contacts', help='path to acronym dataset with Contact-GraspNet folder')
    args = parser.parse_args()
    
    contactnet, optim, config = initialize_net(args.config_path, load_model=True, save_path=args.save_path)
    #from IPython import embed; embed()
    dataset = ContactDataset(args.data_path, config['data'])
    panda_pb = PandaPB(contactnet, dataset, config)
    successful = panda_pb.pb_execute()
