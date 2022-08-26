#!/usr/bin/env python3

import os
import sys
import torch
import pybullet as p
import numpy as np
import lcm
import threading
import rospy
import time
import cv2 as cv
import copy
import random

os.environ["PYOPENGL_PLATFORM"] = "egl"

from franka_interface import ArmInterface

from realsense_lcm.config.default_multi_realsense_cfg import get_default_multi_realsense_cfg
from realsense_lcm.utils.pub_sub_util import RealImageLCMSubscriber, RealCamInfoLCMSubscriber
from realsense_lcm.multi_realsense_publisher_visualizer import subscriber_visualize

cn_path = os.path.join(os.getenv('HOME'), 'graspnet/graspnet/pytorch_contactnet/')
sys.path.append(cn_path)
ik_path = os.path.join(os.getenv('HOME'), 'graspnet/graspnet/pybullet-planning/')
sys.path.append(ik_path)

from model.contactnet import ContactNet
import model.utils.config_utils as config_utils
from train import initialize_loaders, initialize_net
from scipy.spatial.transform import Rotation as R
import argparse
from torch_geometric.nn import fps
from test_meshcat_pcd import viz_pcd as V
from franka_ik import FrankaIK
from simple_multicam import MultiRealsense
#from panda_ndf_utils.panda_mg_wrapper import FrankaMoveIt

class PandaReal():

    def __init__(self, model, config, viz=True):

        rospy.init_node('Panda')
        self.panda = ArmInterface()
        self.panda.set_joint_position_speed(3)
        self.panda.hand.open()
        self.joint_names = self.panda._joint_names
        print('joint names: ', self.joint_names)
        self.ik_helper = FrankaIK(gui=True, base_pos=[0, 0, 0])
        self.model = model
        self.config = config

        #self.renderer = SceneRenderer(intrinsics='realsense')
        self._start_rs_sub()

    def lc_th(self, lc):
        while True:
            lc.handle_timeout(1)
            time.sleep(0.001)
        
    def _start_rs_sub(self):
        '''
        start a thread that runs the LCM subscribers to the realsense camera(s)
        should be called once at the beginning, sets subscribers into self.img_subscribers
        '''
        lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=1")
        rs_cfg = get_default_multi_realsense_cfg()
        serials = rs_cfg.SERIAL_NUMBERS

        serials = [serials[0]] # if multiple cameras are publishing, can pick just one view to use

        rgb_topic_name_suffix = rs_cfg.RGB_LCM_TOPIC_NAME_SUFFIX
        depth_topic_name_suffix = rs_cfg.DEPTH_LCM_TOPIC_NAME_SUFFIX
        info_topic_name_suffix = rs_cfg.INFO_LCM_TOPIC_NAME_SUFFIX
        pose_topic_name_suffix = rs_cfg.POSE_LCM_TOPIC_NAME_SUFFIX

        prefix = rs_cfg.CAMERA_NAME_PREFIX
        camera_names = [f'{prefix}{i}' for i in range(len(serials))]

        # update the topic names based on each individual camera
        rgb_sub_names = [f'{cam_name}_{rgb_topic_name_suffix}' for cam_name in camera_names]
        depth_sub_names = [f'{cam_name}_{depth_topic_name_suffix}' for cam_name in camera_names]
        info_sub_names = [f'{cam_name}_{info_topic_name_suffix}' for cam_name in camera_names]
        pose_sub_names = [f'{cam_name}_{pose_topic_name_suffix}' for cam_name in camera_names]

        self.img_subscribers = []
        for i, name in enumerate(camera_names):
            img_sub = RealImageLCMSubscriber(lc, rgb_sub_names[i], depth_sub_names[i])
            info_sub = RealCamInfoLCMSubscriber(lc, pose_sub_names[i], info_sub_names[i])
            self.img_subscribers.append((name, img_sub, info_sub))

        self.cams = MultiRealsense(camera_names, cfg=None)

        lc_thread = threading.Thread(target=self.lc_th, args=(lc,))
        lc_thread.daemon = True
        lc_thread.start()
        return lc_thread

    def get_images(self):
        '''
        render rgb and depth from cameras.
        important: must call self.start_rs_sub() first
        '''
        self.panda.hand.open()
        time.sleep(1)
        self.reset_joints = list(self.panda.joint_angles().values())
        print('getting images')
        rgb = []
        depth = []
        cam_poses = []
        cam_ints = []
        for (name, img_sub, info_sub) in self.img_subscribers:
            rgb_image, depth_image = img_sub.get_rgb_and_depth(block=True)
            print('rgb and depth got')
            cam_int = info_sub.get_cam_intrinsics(block=True)
            print('camera intrinsics got')
            #cam_pose = info_sub.get_cam_pose()
            depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)
            rgb.append(rgb_image)
            depth.append(depth_image)
            #cam_poses.append(cam_pose)
            cam_ints.append(cam_int)
        return rgb, depth, cam_poses, cam_ints

    def get_pcd(self):
        '''
        convert depth image to pointcloud in world frame, then center
        '''
        rgb, depth, cam_poses = self.get_images()
        camera = cam_poses[0]
        
        # depth to point cloud
        cam_pcd = self.renderer._to_pointcloud(depth[0])
        xr = R.from_euler('x', np.pi, degrees=False)
        x_rot = np.eye(4)
        x_rot[:3, :3] = xr.as_matrix()
        pcd = np.dot(x_rot, cam_pcd)
        pcd = np.dot(camera, pcd).T
        
        # center pcd
        mean = np.mean(pcd[:,:3], axis=0)
        pcd = pcd[:,:3] - mean
        
        # visualize pcd in meshcat (debugging)
        V(pcd, 'camera_pcd', clear=True)

        return pcd

    def get_pcd_ar(self):

        # self.panda.move_to_neutral()
        # time.sleep(5)
        rgb_list, depth_list, cam_pose_list, cam_int_list = self.get_images()

        pcd_pts = []
        for idx, cam in enumerate(self.cams.cams):
            print('getting pcd, camera', cam)
            print(cam_int_list[idx])
            #rgb, depth = img_subscribers[idx][1].get_rgb_and_depth(block=True)
            rgb, depth = rgb_list[idx], depth_list[idx]
            cam_intrinsics = cam_int_list[idx]

            cam.cam_int_mat = cam_intrinsics
            cam._init_pers_mat()
            cam_pose_world = cam.cam_ext_mat

            depth = depth * 0.001
            valid = depth < cam.depth_max
            valid = np.logical_and(valid, depth > cam.depth_min)
            depth_valid = copy.deepcopy(depth)
            depth_valid[np.logical_not(valid)] = 0.0 # not exactly sure what to put for invalid depth

            pcd_cam = cam.get_pcd(in_world=False, filter_depth=False, rgb_image=rgb, depth_image=depth_valid)[0]
            #pcd_world = util.transform_pcd(pcd_cam, cam_pose_world)
            pcd_world = np.matmul(cam_pose_world, np.hstack([pcd_cam, np.ones((pcd_cam.shape[0], 1))]).T).T[:, :-1]
            
            pcd_pts.append(pcd_world)

        pcd_full = np.concatenate(pcd_pts, axis=0)
        x_mask = pcd_full[:,0] > 0.2
        pcd_full = pcd_full[np.nonzero(x_mask)[0], :]
        print('got pcd')
        # from IPython import embed; embed()
        
        return pcd_full

    def infer(self, pcd, threshold=0.6):
        '''
        Forward pass rendered point cloud into the model
        '''
        print('starting inference')
        self.model.eval()
        
        downsample = np.array(random.sample(range(pcd.shape[0]-1), 20000))
        pcd = pcd[downsample, :]
        V(pcd, 'pcd_downsampled', clear=False)

        pcd = torch.Tensor(pcd).to(dtype=torch.float32).to(self.model.device)
        batch = torch.ones(pcd.shape[0]).to(dtype=torch.int64).to(self.model.device)
        idx = fps(pcd, batch, 2048/pcd.shape[0]) #torch.linspace(0, pcd.shape[0]-1, 2048).to(dtype=torch.int64).to(self.model.device)

        points, pred_grasps, pred_successes, pred_widths = self.model(pcd[:, 3:], pos=pcd[:, :3], batch=batch, idx=idx, k=None)

        print('model pass')
        pred_grasps = torch.flatten(pred_grasps, start_dim=0, end_dim=1).detach().cpu().numpy()
        pred_successes = torch.flatten(pred_successes).detach().cpu().numpy()
        pred_widths = torch.flatten(pred_widths, start_dim=0, end_dim=1).detach().cpu().numpy()
        points = torch.flatten(points, start_dim=0, end_dim=1).detach().cpu().numpy()

        success_mask = (pred_successes > threshold).nonzero()

        return pred_grasps[success_mask], pred_successes[success_mask]

    def motion_plan(self, grasp, current_joints):
        '''
        input:
            grasp - a single 4x4 pose of end effector
            current_joints - current pose of the robot
        output:
            joint trajectory for entire motion plan (including pre-grasp pose)
        '''

        # rotate grasp by pi/2 about the z axis (urdf fix)
        z_r = R.from_euler('z', np.pi/2, degrees=False)
        z_rot = np.eye(4)
        z_rot[:3,:3] = z_r.as_matrix()
        z_rot[2,3] += 0.08
        z_rot = np.matmul(z_rot, np.linalg.inv(grasp))
        z_rot = np.matmul(grasp, z_rot)
        grasp = np.matmul(z_rot, grasp)

        offset = np.eye(4)
        offset[2, 3] = -0.1
        offset = np.matmul(offset, np.linalg.inv(grasp))
        offset = np.matmul(grasp, offset)
        pre_grasp = np.matmul(offset, grasp)
        pre_pos = pre_grasp[:3, -1]
        pre_rot = R.from_matrix(pre_grasp[:3, :3])
        pre_quat = pre_rot.as_quat()

        lift_pose = copy.deepcopy(grasp)
        lift_pose[2, 3] += 0.2
        lift_pos = lift_pose[:3, 3]

        rot = R.from_matrix(grasp[:3, :3])
        quat = rot.as_quat()
        pos = grasp[:3, -1]
        
        pose = tuple([*pos, *quat])
        pre_pose = tuple([*pre_pos, *pre_quat])
        lift_pose = tuple([*lift_pos, *quat])
        pre_jnts = self.ik_helper.get_feasible_ik(pre_pose, target_link=True) #??
        sol_jnts = self.ik_helper.get_feasible_ik(pose, target_link=True)
        lift_jnts = self.ik_helper.get_feasible_ik(lift_pose, target_link=True)
        if pre_jnts is None or sol_jnts is None or lift_jnts is None:
            print('having a bad time with IK')
            return None
        
        pre_plan = self.ik_helper.plan_joint_motion(current_joints, pre_jnts)
        sol_plan = self.ik_helper.plan_joint_motion(pre_jnts, sol_jnts)
        lift_plan = self.ik_helper.plan_joint_motion(sol_jnts, lift_jnts)
        reset_plan = self.ik_helper.plan_joint_motion(lift_jnts, self.reset_joints)

        if pre_plan is not None and sol_plan is not None:
            pre_plan = np.concatenate((pre_plan, sol_plan), axis=0)
        else:
            pre_plan = None
        post_plan = np.array(lift_plan)
        reset_plan = np.array(reset_plan)

        #return plan
        if pre_plan is not None and post_plan is not None:
            return (pre_plan, post_plan, reset_plan)
        else:
            return None

    def plan2dict(self, plan):
        dict_plan = [dict(zip(self.joint_names, val.tolist())) for val in plan]
        return dict_plan

    def execute(self, plan):
        # plan is a tuple!

        print('debugging')
        print(plan[0][0])
        print(self.panda.joint_angles())
        from IPython import embed; embed()
        
        self.panda.hand.open()
        s0 = self.panda.execute_position_path(self.plan2dict(plan[0]))
        # from IPython import embed; embed()
        self.panda.hand.close()
        s1 = self.panda.execute_position_path(self.plan2dict(plan[1]))
        time.sleep(5)
        val = input('press enter to reset')
        self.panda.hand.open()
        s2 = self.panda.execute_position_path(self.plan2dict(plan[2]))
        #self.panda.move_to_neutral()
        print('execute attempt:', s0, s1)
        return (s0, s1)

    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', type=bool, default=True, help='whether or not to debug visualize in meshcat')
    parser.add_argument('--load_path', type=str, default='./checkpoints/train_10/epoch_62.pth', help='path to load model from')
    parser.add_argument('--config_path', type=str, default='./model/', help='path to config yaml file')
    args = parser.parse_args()

    # Initialize model, initialize robot stuff
    contactnet, optim, config = initialize_net(args.config_path, load_model=True, save_path=args.load_path)
    panda_robot = PandaReal(contactnet, config, args.viz)
    #panda_moveit = FrankaMoveIt(panda_robot.panda)
    # Get pcd, pass into model
    pcd = panda_robot.get_pcd_ar()
    pred_grasps, pred_success = panda_robot.infer(pcd, threshold=0.05)
    plan = None
    V(pcd, 'pcd', clear=True)
    for i, grasp in enumerate(pred_grasps):
        g_target = np.eye(4)
        g_target[2, 3] = 0.05
        grasp = np.matmul(grasp, g_target)
        #V(grasp, f'gripper_{i}', gripper=True)
        V(grasp, f'gripper', gripper=True)
        try_execute = input("enter y to plan, else skip: ")
        if try_execute == 'y':
            current_joints = panda_robot.panda.joint_angles()
            plan = panda_robot.motion_plan(grasp, list(current_joints.values()))
            if plan is not None:
                for jpos in plan[0]:
                    panda_robot.ik_helper.set_jpos(jpos)
                    time.sleep(0.1)
                time.sleep(0.5)
                for jpos in plan[1]:
                    panda_robot.ik_helper.set_jpos(jpos)
                    time.sleep(0.1)

                execute = input("enter y to run execute, else skip: ")
                if execute == 'y':
                    success = panda_robot.execute(plan)
        else:
            print('skipping')
        from IPython import embed; embed()
        #test
