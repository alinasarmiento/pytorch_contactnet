import os
import os.path
import sys
import numpy as np
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from torch.utils.data import DataLoader, Dataset
from keras.utils.np_utils import to_categorical
# TO-DO: put import of dataset here

from pointnet2.models_pointnet import FPModule, SAModule, MLP
import utils.pcd_utils as utils

class ContactNet(nn.Module):
    def __init__(self, generate, latent_size, device, config):
        super().__init__()
        self.device = device
        self.config = config
        self.setabstract = self.SAnet(config.sa)
        self.featprop = self.FPnet(config.fp)
        self.multihead = self.Multihead(config.multi)
        
    def forward(self, inputpcd, k=2048):
        '''
        maps each point in the pointcloud to a generated grasp
        Arguments
            inputpcd (torch.Tensor): full pointcloud of cluttered scene
            k (int): number of points in the pointcloud to downsample to and generate grasps for (if None, use all points)
        Returns
            list of grasps (4x4 numpy arrays)
        '''
        # sample points on the pointcloud to generate grasps from
        sample_pts = inputpcd
        if k is not None:
            sample_pts = utils.farthest_point_downsample(inputpcd, k)
        
        # pass sampled points through network to generate grasps
        latent = self.setabstract(sample_pts) #high dimensional, sparse set abstraction
        ptfeats = self.featprop(latent) #pointwise features
        finals = []
        for net in self.multihead:
            finals.append(net(ptfeats))
        s, z1, z2, w = finals # confidence prediction, grasp vector 1, grasp vector 2, grasp width

        # build final grasps
        final_grasps = self.build_6d_grasps(sample_pts, z1, z2, w, self.config.gripper_depth)

        return final_grasps, s
        
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

    def lossfn(self, output, labels):
        '''
        loss function
        '''
        # TO-DO
        
    def SAnet(self, cfg):
        '''
        part of the net that downsizes the pointcloud
        
        cfg: config dict
            radii - list of radii for each level
            centers - list of number of neighborhoods to sample for each level
            mlps - list of lists of mlp layers for each level, first mlp must start with in_dimension
        '''
        samodules = nn.ModuleList()
        for r, centers, mlplist in zip(cfg.radii, cfg.centers, cfg.mlps):
            module = SAModule(r, centers, MLP(mlplist))
            samodules.append(module)
        return samodules
        
    def FPnet(self, cfg):
        '''
        part of net that upsizes the pointcloud

        cfg: config dict
            klist - list of k nearest neighbors to interpolate between
            nnlist - list of unit pointclouds to run between feat prop layers
        '''
        fpmodules = nn.ModuleList()
        for k, nn in zip(klist, nnlist):
            module = FPModule(k, nn)
            fpmodules.append(module)
        return fpmodules

    def multihead(self, cfg):
        '''
        four multihead net from feature propagation, creates final predictions

        cfg: config dict
            pointnetout - dimension of output of pointnet (2048)
            outdims - list of output dimensions for each head
            ps - list of dropout rates for each head
        note: heads are listed in order SUCCESS_CONFIDENCE, Z1, Z2, WIDTH
        '''
        headlist = []
        for outdim, p in zip(cfg.outdims, cfg.ps):
            head = nn.Sequential(nn.Conv1d(cfg.pointnetout, 128, 1), nn.Dropout(p), nn.Conv1d(128, outdim))
            headlist.append(head)
        return headlist
