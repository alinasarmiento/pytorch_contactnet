#!/usr/bin/env python3

import os
import sys
import torch
import numpy as np
import copy
import random
import meshcat
cn_path = os.path.join(os.getenv('HOME'), 'pytorch_contactnet')
sys.path.append(cn_path)

from model.contactnet import ContactNet
import model.utils.config_utils as config_utils
import model.utils.mesh_utils as mesh_utils
import argparse
from torch_geometric.nn import fps
from test_meshcat_pcd import meshcat_pcd_show as viz_points
from test_meshcat_pcd import sample_grasp_show as viz_grasps
from test_meshcat_pcd import show_mesh
from dataset import get_dataloader
from scipy.spatial.transform import Rotation as R
import trimesh

def initialize_net(config_file, load_model, save_path, args):
    print('initializing net')
    torch.cuda.empty_cache()
    config_dict = config_utils.load_config(config_file)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ContactNet(config_dict, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if load_model==True:
        print('loading model')
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer, config_dict

def initialize_loaders(data_pth, data_config, batch, size=None, include_val=False, val_path=None, preloaded=False, args=None):
    print('initializing loaders')
    if batch==None:
        batch = data_config['batch_size']
    train_loader = get_dataloader(data_pth, batch, size=size, data_config=data_config, preloaded=preloaded, args=args)
    print('train loader got.')
    return train_loader

def grasp_to_gripper(grasp_pose, translate=0.0, theta=np.pi/2):
    '''                                                                                                                                                     
    does a small conversion between the grasp frame and the actual gripper frame for IK                                                                     
    '''
    z_rot = np.eye(4)
    z_rot[2,3] = translate
    z_rot[:3,:3] = R.from_euler('z', theta).as_matrix()
    z_tf = np.matmul(z_rot, np.linalg.inv(grasp_pose))
    z_tf = np.matmul(grasp_pose, z_tf)
    gripper_pose = np.matmul(z_tf, grasp_pose)

    return gripper_pose


def cgn_infer(cgn, pcd, obj_mask=None, threshold=0.5):
    cgn.eval()
    if pcd.shape[0] > 20000:
        downsample = np.array(random.sample(range(pcd.shape[0]-1), 20000))
    else:
        downsample = np.arange(20000)
    pcd = pcd[downsample, :]

    pcd = torch.Tensor(pcd).to(dtype=torch.float32).to(cgn.device)
    batch = torch.zeros(pcd.shape[0]).to(dtype=torch.int64).to(cgn.device)
    idx = fps(pcd, batch, 2048/pcd.shape[0])
    #idx = torch.linspace(0, pcd.shape[0]-1, 2048).to(dtype=torch.int64).to(cgn.device)
    
    if obj_mask is not None:
        obj_mask = torch.Tensor(obj_mask[downsample])
        obj_mask = obj_mask[idx]
    else:
        obj_mask = torch.ones(idx.shape[0])

    points, pred_grasps, confidence, pred_widths, _, pred_collide = cgn(pcd[:, 3:], pos=pcd[:, :3], batch=batch, idx=idx, obj_mask=[obj_mask])
    sig = torch.nn.Sigmoid()
    confidence = sig(confidence)
    confidence = confidence.reshape(-1,1)
    pred_grasps = torch.flatten(pred_grasps, start_dim=0, end_dim=1).detach().cpu().numpy()

    confidence = (obj_mask.detach().cpu().numpy() * confidence.detach().cpu().numpy()).reshape(-1)
    pred_widths = torch.flatten(pred_widths, start_dim=0, end_dim=1).detach().cpu().numpy()
    points = torch.flatten(points, start_dim=0, end_dim=1).detach().cpu().numpy()

    success_mask = (confidence > threshold).nonzero()[0]
    if len(success_mask) == 0:
        print('failed to find successful grasps')
        raise Exception

    return pred_grasps[success_mask], confidence[success_mask], downsample

def visualize(pcd, grasps, mc_vis=None):
    if mc_vis is None:
        mc_vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
    viz_points(mc_vis, pcd, name='pointcloud', color=(0,0,0), size=0.002)
    grasp_kp = get_key_points(grasps)
    viz_grasps(mc_vis, grasp_kp, name='gripper/', freq=1)
    
def get_key_points(poses, include_sym=False):
    gripper_object = mesh_utils.create_gripper('panda', root_folder=cn_path)

    gripper_np = gripper_object.get_control_point_tensor(poses.shape[0])
    hom = np.ones((gripper_np.shape[0], gripper_np.shape[1], 1))
    gripper_pts = np.concatenate((gripper_np, hom), 2).transpose(0,2,1)
    pts = np.matmul(poses, gripper_pts).transpose(0,2,1)

    return pts
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz', type=bool, default=True, help='whether or not to debug visualize in meshcat')
    parser.add_argument('--load_path', type=str, default='./checkpoints/current.pth', help='path to load model from')
    parser.add_argument('--config_path', type=str, default='./model/', help='path to config yaml file')
    parser.add_argument('--model', type=str, default='sg_score')
    parser.add_argument('--pos_weight', default=1.0)
    parser.add_argument('--threshold', default=0.9, type=float, help='success threshold for grasps')
    parser.add_argument('--scene', default='005274.npz', help='file with demo scene loaded')
    args = parser.parse_args()

    ### Initialize net
    contactnet, optim, config = initialize_net(args.config_path, load_model=True, save_path=args.load_path, args=args)
    
    ### Get demo pointcloud
    print('getting demo')
    arrays = np.load(args.scene, allow_pickle=True)
    [k0, k1, k2, k3, k4, k5] = arrays.files
    pcd_list, _, obj_mask, mean, cam_pose, _ = arrays[k0], arrays[k1], arrays[k2], arrays[k3], arrays[k4], arrays[k5]
    pointcloud = pcd_list[0]
    
    ### Get pcd, pass into model
    print('inferring.')
    pred_grasps, pred_success, downsample = cgn_infer(contactnet, pointcloud, obj_mask, threshold=args.threshold)
    print('model pass.', pred_grasps.shape[0], 'grasps found.')
    
    ### Visualize
    visualize(pointcloud, pred_grasps)
