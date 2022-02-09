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
import model.utils.pcd_utils as utils
import model.utils.mesh_utils as mesh_utils
sys.path.append('../pointnet2')
from pointnet2.models_pointnet import FPModule, SAModule, MLP


class ContactNet(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.set_abstract = self.SAnet(config['model']['sa'])
        self.feat_prop = self.FPnet(config['model']['fp'])
        self.multihead = self.Multihead(config['model']['multi'])
        
    def forward(self, input_pcd, pos, batch, k=None):
        '''
        maps each point in the pointcloud to a generated grasp
        Arguments
            input_pcd (torch.Tensor): full pointcloud of cluttered scene
            k (int): number of points in the pointcloud to downsample to and generate grasps for (if None, use all points)
        Returns
            list of grasps (4x4 numpy arrays)
        '''
        np.save('first_pcd', pos.cpu().numpy())
        sample_pts = input_pcd.float()
        if k is not None:
            sample_pts = utils.farthest_point_downsample(input_pcd, k)
        input_list = (sample_pts, pos, batch)

        skip_layers = [input_list]
        for module_list in self.set_abstract:
            
            feature_cat = torch.Tensor().to(self.device)
            for i, module in enumerate(module_list):
                
                if i==0:    
                    feat, pos, batch, idx = module(*input_list)
                    
                else:
                    feat, pos, batch, idx = module(*input_list, sample=False, idx=idx)
                    np.save('pts_check', pos.cpu().numpy())
                feature_cat = torch.cat((feature_cat, feat), 1)
            
            input_list = (feature_cat, pos, batch)
            skip_layers.insert(0, input_list)
            

        for module, skip in zip(self.feat_prop, skip_layers):
            #print('MODULE FEAT PROP')
            input_list = module(*input_list, *skip)
        point_features =  input_list[0]  #pointwise features
        
        points = input_list[1]
        
        point_features = torch.cat((points, point_features), 1)
        point_features = torch.unsqueeze(point_features, 0)
        point_features = point_features.transpose(1, 2)
        # feed into final multihead
        finals = []
        for net in self.multihead:
            result = torch.flatten(net(point_features).transpose(1,2), start_dim=0, end_dim=1)
            finals.append(result)
        
        s, z1, z2, w = finals # confidence prediction, grasp vector 1, grasp vector 2, grasp width
        
        # build final grasps
        final_grasps = self.build_6d_grasps(points, z1, z2, w, self.config['gripper_depth'])
        #final_grasps = torch.Tensor(final_grasps)
        # unsqueeze point features and points into batches
        num_batches = int(torch.max(input_list[2]))+1
        pts_shape = (num_batches, input_list[1].shape[0]//num_batches, -1)
        grasp_shape = (num_batches, input_list[1].shape[0]//num_batches, 4, 4)
        scalar_shape = (num_batches, input_list[1].shape[0]//num_batches)
        points = input_list[1].view(pts_shape).to(self.device)
        final_grasps = final_grasps.view(grasp_shape).to(self.device)
        success_sigmoid = nn.Sigmoid()
        width_sigmoid = nn.Sigmoid()
        s = success_sigmoid(s).to(self.device)
        w = width_sigmoid(w).to(self.device)

        s = s.view(scalar_shape).to(self.device)
        w = w.view(scalar_shape).to(self.device)
        
        return points, final_grasps, s, w
                
    def filter_segment(self, seg_mask, grasps):
        '''
        filters out grasps to just the provided segmentation mask
        '''
        filtered_grasps = grasps[seg_mask]
        return filtered_grasps

    def build_6d_grasps(self, contact_pts, z1, z2, w, gripper_depth):
        '''
        builds full 6 dimensional grasps based on generated vectors, width, and pointcloud
        '''
        # calculate baseline and approach vectors based on z1 and z2
        #contact_pts = contact_pts.cpu()#.detach().numpy()
        #z1 = z1.cpu()#.detach().numpy()
        #z2 = z2.cpu()#.detach().numpy()
        #w = w.cpu()#.detach().numpy()
        
        base_dirs = z1/(torch.unsqueeze(torch.linalg.norm(z1, dim=1), dim=0).transpose(0, 1)) #z1/torch.linalg.norm(z1, dim=1)
        inner = torch.sum((base_dirs * z2), dim=1)
        prod = torch.unsqueeze(inner, -1)*base_dirs
        approach_dirs = (z2 - prod)/(torch.unsqueeze(torch.linalg.norm(z2, dim=1), dim=0).transpose(0, 1))
        #from IPython import embed
        #embed()

        np.save('6d_grasp_building', contact_pts.cpu().numpy())
        
        # build grasps
        grasps = []
        for i in range(len(contact_pts)):
            grasp = torch.eye(4)
            grasp[:3,0] = base_dirs[i] / torch.linalg.norm(base_dirs[i])
            grasp[:3,2] = approach_dirs[i] / torch.linalg.norm(approach_dirs[i])
            grasp_y = torch.cross( grasp.clone()[:3,2],grasp.clone()[:3,0])
            grasp[:3,1] = grasp_y / torch.linalg.norm(grasp_y)
            grasp[:3,3] = contact_pts[i] #+ (w[i] / 2) * grasp.clone()[:3,0].to(self.device) - gripper_depth * grasp.clone()[:3,2].to(self.device)
            grasps.append(grasp)
        grasps = torch.stack(grasps).to(self.device)
        return grasps


    def loss(self, pred_grasps, pred_success, pred_width, labels_dict, args):
        '''
        returns loss as described in original paper
        
        labels_dict
            success (boolean)
            grasps (6d grasps)
        '''
        success_idxs = labels_dict['success_idxs']
        success_labels = np.array(labels_dict['success'])
        grasp_labels = labels_dict['grasps']
        width_labels = labels_dict['width']
        success_labels = success_labels.reshape(success_labels.shape[0], -1) #np.transpose(success_labels, axes=(0,2,1))
        
        success_labels = torch.Tensor(success_labels).to(self.device)            
        
        # -- GRASP CONFIDENCE LOSS --
        # Use binary cross entropy loss on predicted success to find top-k preds with largest loss
        conf_loss_fn = nn.BCELoss(reduction='none').to(self.device)
        conf_loss = torch.mean(torch.topk(conf_loss_fn(pred_success, success_labels), k=512)[0]).to(self.device)

        # -- GEOMETRIC LOSS --
        # Turn each gripper control point into a pose object
        
        # Find positive grasp labels and corresponding predictions
        grasp_labels = grasp_labels.float().to(self.device) #.view(grasp_labels.shape[0], -1, 4, 4).to(self.device)
        pos_label_list = []
        pos_pred_list = []
        # pos_pred - list of predicted grasp poses for positively labeled contact points
        # pos_labels - list of labeled grasp poses for positively labeled contact points

        for batch, idx_list in enumerate(success_idxs):
            idx_list = idx_list.T
            point_idxs = idx_list[0]
            label_idxs = idx_list[1]
            pos_labels = grasp_labels[batch, label_idxs, :4, :4]
            # !! pos_labels[batch, point_idxs, :4, :4] = grasp_labels[0, label_idxs, :4, :4]
            pos_pred = pred_grasps[batch, point_idxs, :4, :4]

            pos_label_list.append(pos_labels)
            pos_pred_list.append(pos_pred)

        label_pts_list = []
        pred_pts_list = []

        gripper_object = mesh_utils.create_gripper('panda', root_folder=args.root_path)

        for pos_labels, pos_pred in zip(pos_label_list, pos_pred_list):

            gripper_np = gripper_object.get_control_point_tensor(pos_labels.shape[0])
            sym_gripper_np = gripper_object.get_control_point_tensor(pos_labels.shape[0], symmetric=True)
            hom = np.ones((gripper_np.shape[0], gripper_np.shape[1], 1))
            gripper_pts = torch.Tensor(np.concatenate((gripper_np, hom), 2)).transpose(1,2).to(self.device)
            sym_gripper_pts = torch.Tensor(np.concatenate((sym_gripper_np, hom), 2)).transpose(1,2).to(self.device)

            label_pts = torch.matmul(pos_labels, gripper_pts).transpose(1,2)
            sym_label_pts = torch.matmul(pos_labels, sym_gripper_pts).transpose(1,2)
            
            pred_pts = torch.matmul(pos_pred, gripper_pts).transpose(1,2)
            
            label_pts_list.append([label_pts, sym_label_pts])
            pred_pts_list.append(pred_pts)


        pred_pts1 = pred_pts_list[0][:,:,:3]
        pred_pts2 = pred_pts_list[1][:,:,:3]
        np.save('control_pt_1', torch.flatten(pred_pts1.detach(), start_dim=0, end_dim=1).cpu().numpy())
        np.save('control_pt_2', torch.flatten(pred_pts2.detach(), start_dim=0, end_dim=1).cpu().numpy())
        # Compare symmetric predicted and label control points to calculate "add-s" loss
        for success_idx, pred_success_list, pred_pts, label_pts in zip(success_idxs, pred_success, pred_pts_list, label_pts_list):
            point_success_mask = success_idx[:, 0]
            pred_pts = pred_pts[:, :, :3]
            label_pts_1 = label_pts[0][:, :, :3]
            label_pts_2 = label_pts[1][:, :, :3]
            
            pred_success_masked = pred_success_list[point_success_mask]
            
            norm_1 = torch.linalg.vector_norm((pred_pts - label_pts_1), dim=(1,2))
            norm_2 = torch.linalg.vector_norm((pred_pts - label_pts_2), dim=(1,2))
            min_norm = torch.min(norm_1, norm_2)
            add_s_loss = pred_success_masked*min_norm
                                    
            add_s_loss = torch.mean(add_s_loss).to(self.device)
        
        # -- APPROACH AND BASELINE LOSSES --
        # currently not used
        
        # -- WIDTH LOSS --
        # calculates loss on 10 grasp width bins using a sigmoid fn and binary cross entropy
        width_labels = torch.Tensor(np.array(width_labels)).to(self.device)
        width_labels = width_labels.reshape(width_labels.shape[0], -1)

        width_loss_fn = nn.MSELoss().to(self.device)
        raw_width_loss = width_loss_fn(pred_width, width_labels)#nn.BCEWithLogitsLoss(pred_width, width_labels)
        
        #masked_width_loss = torch.index_select((pred_success*raw_width_loss), 1, success_idxs)
        width_loss = raw_width_loss #np.mean(masked_width_loss)

        total_loss = self.config['loss']['conf_mult']*conf_loss + self.config['loss']['add_s_mult']*add_s_loss + self.config['loss']['width_mult']*width_loss*0
        print(conf_loss.item(), add_s_loss.item(), width_loss.item())

        
        return conf_loss, add_s_loss, width_loss, total_loss #, approach_loss, baseline_loss
        
    def SAnet(self, cfg):
        '''
        part of the net that downsizes the pointcloud
        
        cfg: config dict
            radii - nested list of radii for each level
            centers - list of number of neighborhoods to sample for each level
            mlps - list of lists of mlp layers for each level, first mlp must start with in_dimension
        '''
        sa_modules = [] #nn.ModuleList()
        input_size = 0
        num_points = 20000
        for r_list, center, mlp_list in zip(cfg['radii'], cfg['centers'], cfg['mlps']):
            layer_modules = []
            feat_cat_size = 0
            input_size += 3
            for r, mlp_layers in zip(r_list, mlp_list):
                mlp_layers.insert(0, input_size)
                module = SAModule((center/num_points), r, MLP(mlp_layers)).to(self.device)
                layer_modules.append(copy.deepcopy(module))
                feat_cat_size += mlp_layers[-1] #keep track of how big the concatenated feature vector is for MSG
            num_points = center
            input_size = feat_cat_size
            sa_modules.append(nn.ModuleList(copy.deepcopy(layer_modules)))
        return sa_modules
        
    def FPnet(self, cfg):
        '''
        part of net that upsizes the pointcloud

        cfg: config dict
            klist - list of k nearest neighbors to interpolate between
            nnlist - list of unit pointclouds to run between feat prop layers
        '''
        fp_modules = nn.ModuleList()
        for k, layer_list in zip(cfg['klist'], cfg['nnlist']):
            module = FPModule(k, MLP(layer_list))
            fp_modules.append(module)
        return fp_modules

    def Multihead(self, cfg):
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
            head = nn.Sequential(nn.Conv1d(cfg['pointnet_out_dim'], 128, 1),
                                 nn.BatchNorm1d(128),
                                 nn.Dropout(p),
                                 nn.Conv1d(128, out_dim, 1)).to(self.device)
            head_list.append(head)
        return head_list
