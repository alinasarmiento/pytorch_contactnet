import os
import os.path
import sys
import numpy as np
import math
import random
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from torch.utils.data import DataLoader, Dataset
from keras.utils.np_utils import to_categorical
# TO-DO: put import of dataset here

from pointnet2.models_pointnet import FPModule, SAModule, MLP
import utils.pcd_utils as utils
import utils.mesh_utils as mesh_utils

class ContactNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.set_abstract = self.SAnet(config.sa)
        self.feat_prop = self.FPnet(config.fp)
        self.multihead = self.Multihead(config.multi)
        
    def forward(self, input_pcd, k=None):
        '''
        maps each point in the pointcloud to a generated grasp
        Arguments
            input_pcd (torch.Tensor): full pointcloud of cluttered scene
            k (int): number of points in the pointcloud to downsample to and generate grasps for (if None, use all points)
        Returns
            list of grasps (4x4 numpy arrays)
        '''
        # sample points on the pointcloud to generate grasps from
        sample_pts = input_pcd
        if k is not None:
            sample_pts = utils.farthest_point_downsample(input_pcd, k)
        
        # pass sampled points through network to generate grasps
        latent = self.set_abstract(sample_pts) #high dimensional, sparse set abstraction
        point_features = self.feat_prop(latent) #pointwise features
        finals = []
        for net in self.multihead:
            finals.append(net(point_features))
        s, z1, z2, w = finals # confidence prediction, grasp vector 1, grasp vector 2, grasp width

        # build final grasps
        final_grasps = self.build_6d_grasps(sample_pts, z1, z2, w, self.config.gripper_depth)

        return final_grasps, s, w
        
    def build_6d_grasps(self, contact_pts, z1, z2, w, gripper_depth):
        '''
        builds full 6 dimensional grasps based on generated vectors, width, and pointcloud
        '''
        # calculate baseline and approach vectors based on z1 and z2
        base_dirs = z1/np.linalg.norm(z1)
        approach_dirs = (z2 - (np.dot(base_dirs, z2)*base_dirs))/np.linalg.norm(z2)
        
        # build grasps
        grasps = []
        for i in range(len(contact_pts)):
            grasp = np.eye(4)
            grasp[:3,0] = base_dirs[i] / np.linalg.norm(base_dirs[i])
            grasp[:3,2] = approach_dirs[i] / np.linalg.norm(approach_dirs[i])
            grasp_y = np.cross( grasp[:3,2],grasp[:3,0])
            grasp[:3,1] = grasp_y / np.linalg.norm(grasp_y)
            grasp[:3,3] = contact_pts[i] + w[i] / 2 * grasp[:3,0] - gripper_depth * grasp[:3,2]
            grasps.append(grasp)

        return np.array(grasps)
        
    def filter_segment(self, seg_mask, grasps):
        '''
        filters out grasps to just the provided segmentation mask
        '''
        filtered_grasps = grasps[seg_mask]
        return filtered_grasps

    def loss(self, pred_grasps, pred_success, pred_width, labels_dict):
        '''
        returns loss as described in original paper
        
        labels_dict
            success (boolean)
            grasps (6d grasps)
        '''
        success_labels = labels_dict['success']
        grasp_labels = labels_dict['grasps']
        width_labels = labels_dict['width']

        # -- GRASP CONFIDENCE LOSS --
        # Use binary cross entropy loss on predicted success to find top-k preds with largest loss
        conf_loss = nn.BCELoss(pred_success, success_labels)

        # -- GEOMETRIC LOSS --
        # Turn each gripper control point into a pose object
        gripper_np = mesh_utils.create_gripper('panda')
        gripper_poses = []
        for point in gripper_np:
            pt_pose = utils.list2pose_stamped(np.concatenate(point, [0, 0, 0, 1]))
            gripper_poses.append(pt_pose)

        # Find positive grasp labels and corresponding predictions
        pos_pred_mask = np.where(grasp_labels>0)
        pos_labels = grasp_labels[np.where(pos_pred_mask)]
        pos_pred = pred_grasps[np.where(pos_pred_mask)]

        # Turn positive predicted grasp labels into gripper control points
        label_control_points = []
        pred_control_points = []
        for i in range(len(pos_grasp_labels)):
            pred_pose = utils.pose_from_matrix(pos_pred[i])
            label_pose = utils.pose_from_matrix(pos_labels[i])
            pred_pts = []
            label_pts = []
            # Convert each point in the gripper control points to the predicted and label poses
            for pt_pose in gripper_poses:
                pred_pt = utils.convert_reference_frame(copy.deepcopy(pt_pose), utils.unit_pose, pred_pose)
                pred_pts.append([pred_pt.pose.position.x, pred_pt.pose.position.y, pred_pt.pose.position.z])
                label_pt = utils.convert_reference_frame(copy.deepcopy(pt_pose), utils.unit_pose(), labels_pose)
                label_pts.append([label_pt.pose.position.x, label_pt.pose.position.y, label_pt.pose.position.z])
            pred_control_points.append(pred_pts)
            label_control_points.append(label_pts)

        # Compare symmetric predicted and label control points to calculate "add-s" loss
        add_s_loss = np.mean(pred_succcess*(np.min(np.linalg.norm(pred_control_points - label_control_points), 0)))
        
        # -- APPROACH AND BASELINE LOSSES --
        #currently not used
        
        # -- WIDTH LOSS --
        # calculates loss on 10 grasp width bins using a sigmoid fn and binary cross entropy
        raw_width_loss = nn.BCEWithLogitsLoss(pred_width, width_labels)
        masked_width_loss = (pred_success*raw_width_loss)[np.where(success_labels)]
        width_loss = np.mean(masked_width_loss)

        total_loss = self.config['loss']['conf_mult']*conf_loss + self.config['loss']['add_s_mult']*add_s_loss + self.config['loss']['width_mult']*width_loss
    
        return conf_loss, add_s_loss, width_loss, total_loss #, approach_loss, baseline_loss
        
    def SAnet(self, cfg):
        '''
        part of the net that downsizes the pointcloud
        
        cfg: config dict
            radii - nested list of radii for each level
            centers - list of number of neighborhoods to sample for each level
            mlps - list of lists of mlp layers for each level, first mlp must start with in_dimension
        '''
        sa_modules = nn.ModuleList()
        for r, centers, mlp_list in zip(cfg['radii'], cfg['centers'], cfg['mlps']):
            module = SAModule(r, centers, MLP(mlp_list))
            sa_modules.append(module)
        return sa_modules
        
    def FPnet(self, cfg):
        '''
        part of net that upsizes the pointcloud

        cfg: config dict
            klist - list of k nearest neighbors to interpolate between
            nnlist - list of unit pointclouds to run between feat prop layers
        '''
        fp_modules = nn.ModuleList()
        for k, nn in zip(cfg['klist'], cfg['nnlist']):
            module = FPModule(k, MLP(nn))
            fp_modules.append(module)
        return fp_modules

    def multihead(self, cfg):
        '''
        four multihead net from feature propagation, creates final predictions

        cfg: config dict
            pointnetout - dimension of output of pointnet (2048)
            outdims - list of output dimensions for each head
            ps - list of dropout rates for each head
        note: heads are listed in order SUCCESS_CONFIDENCE, Z1, Z2, WIDTH
        '''
        head_list = []
        for out_dim, p in zip(cfg['out_dims'], cfg['ps']):
            head = nn.Sequential(nn.Conv1d(cfg['pointnet_out_dim'], 128, 1), nn.Dropout(p), nn.Conv1d(128, out_dim))
            head_list.append(head)
        return head_list
