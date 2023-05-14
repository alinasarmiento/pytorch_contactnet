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
from torch_geometric.nn import fps, knn_interpolate
from torchvision import transforms, utils
from torch.utils.data import DataLoader, Dataset
import model.utils.pcd_utils as utils
import model.utils.mesh_utils as mesh_utils
sys.path.append('../pointnet2')
from pointnet2.models_pointnet import FPModule, SAModule, MLP
from test_meshcat_pcd import viz_pcd as V
from data_utils import get_obj_surrounding

class ContactNet(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.feat_cat_list = [0]
        self.set_abstract_msg = self.SAnet_msg(config['model']['sa']) #set abstraction with multi-scale grouping
        self.set_abstract_final = self.SAnet(config['model']['sa_final']) #final set abstraction model without multi-scale grouping
        self.feat_prop = self.FPnet(config['model']['fp']) #feature propagation 
        self.multihead = self.Multihead(config['model']['multi']) #final multihead for 2 vectors, confidence, and width

        self.success_sigmoid = nn.Sigmoid().to(self.device)
        self.width_relu = nn.ReLU().to(self.device)
        
        self.conf_loss_fn = nn.BCEWithLogitsLoss(reduction='none').to(self.device) #nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([1])).to(self.device)
        
    def forward(self, input_pcd, pos, batch, idx, width_labels=None):
        '''
        maps each point in the pointcloud to a generated grasp
        Arguments
            input_pcd (torch.Tensor): full pointcloud of cluttered scene
            k (int): number of points in the pointcloud to downsample to and generate grasps for (if None, use all points)
        Returns
            list of grasps (4x4 numpy arrays)
        '''
        # Step 1: PointNet++ Set Aggregation (with multi-scale grouping)

        sample_pts = input_pcd.float()
        input_list = (sample_pts, pos, batch)
        skip_layers = [input_list]
        for mod_idx, module_list in enumerate(self.set_abstract_msg):
            feature_cat = torch.Tensor().to(self.device)
            for i, module in enumerate(module_list):
                if (i==0) and (mod_idx!=0):    # sample down point list (beginning of every module except very first one in network)
                    feat, pos, batch, idx = module(*input_list)
                else:
                    feat, pos, batch, idx = module(*input_list, sample=False, idx=idx)
                feature_cat = torch.cat((feature_cat, feat), 1)
            input_list = (feature_cat, pos, batch)
            skip_layers.insert(0, input_list)
        # final SA module (no multi-scale grouping)
        feat = self.set_abstract_final(input_list[0])
        skip_layers.insert(0, (feat, input_list[1], input_list[2]))

        # Step 2: PointNet++ Feature Propagation

        for module, skip in zip(self.feat_prop, skip_layers[1:]):
            input_list = module(*input_list, *skip)
        point_features =  input_list[0]  #pointwise features
        points = input_list[1]
        point_features = torch.cat((points, point_features), 1)
        point_features = torch.unsqueeze(point_features, 0)
        point_features = point_features.transpose(1, 2)
        
        # Step 3: Final Multihead Output

        finals = []
        for net in self.multihead[1:]:
            result = torch.flatten(net(point_features).transpose(1,2), start_dim=0, end_dim=1)
            finals.append(result)
        z1, z2, w = finals # confidence prediction, grasp vector 1, grasp vector 2, grasp width

        z1 = z1/torch.linalg.norm(z1, dim=1, keepdim=True)
        z2 = z2 - torch.sum(z1*z2, dim=-1, keepdim=True)*z1
        z2 = z2/torch.linalg.norm(z2, dim=1, keepdim=True)
        w = torch.clamp(w, min=-0.08, max=0.08)
        
        # build final grasps
        final_grasps = self.build_6d_grasps(points, z1, z2, w, self.config['gripper_depth'], width_labels)

        # unsqueeze point features and points into batches
        num_batches = int(torch.max(input_list[2]))+1
        pts_shape = (num_batches, input_list[1].shape[0]//num_batches, -1)
        grasp_shape = (num_batches, input_list[1].shape[0]//num_batches, 4, 4)
        scalar_shape = (num_batches, input_list[1].shape[0]//num_batches)

        points = input_list[1].view(pts_shape).to(self.device)
        final_grasps = final_grasps.view(grasp_shape).to(self.device)
        s = torch.flatten(self.multihead[0](point_features).transpose(1,2), start_dim=0, end_dim=1)
        
        # s = self.success_sigmoid(s) #.to(self.device)
        w = self.width_relu(w)#.to(self.device)
        s = s.view(scalar_shape)#.to(self.device)
        w = w.view(scalar_shape)#.to(self.device)
        point_features = point_features.transpose(1,2)
        point_features = point_features.view(num_batches, -1, 131)
        points = points.view(num_batches, -1, 3)

        collide_pred_list = None
        
        return points, final_grasps, s, w, point_features, collide_pred_list
                
    def build_6d_grasps(self, contact_pts, z1, z2, w, gripper_depth=0.1034, width_labels=None):
        '''
        builds full 6 dimensional grasps based on generated vectors, width, and pointcloud
        '''
        # calculate baseline and approach vectors based on z1 and z2
        base_dirs = z1/(torch.unsqueeze(torch.linalg.norm(z1, dim=1), dim=0).transpose(0, 1)) #z1/torch.linalg.norm(z1, dim=1)
        inner = torch.sum((base_dirs * z2), dim=1)
        prod = torch.unsqueeze(inner, -1)*base_dirs
        approach_dirs = (z2 - prod)/(torch.unsqueeze(torch.linalg.norm(z2, dim=1), dim=0).transpose(0, 1))

        # build grasps
        grasps = []
        for i in range(len(contact_pts)):
            grasp = torch.eye(4)
            grasp[:3,0] = base_dirs[i] / torch.linalg.norm(base_dirs[i])
            grasp[:3,2] = approach_dirs[i] / torch.linalg.norm(approach_dirs[i])
            grasp_y = torch.cross( grasp.clone()[:3,2],grasp.clone()[:3,0])
            grasp[:3,1] = grasp_y / torch.linalg.norm(grasp_y)

            grasp[:3,3] = contact_pts[i] - gripper_depth*grasp.clone()[:3,2].to(self.device) + (w[i]/2)*grasp.clone()[:3,0].to(self.device)

            if torch.linalg.norm(grasp[:3,3]) > 100:
                print('grasp building issue')
                from IPython import embed; embed()
            grasps.append(grasp)
            
        grasps = torch.stack(grasps).to(self.device)
        return grasps

    def get_key_points(self, grasps_list, include_sym=False):
        pts_list = []
        gripper_object = mesh_utils.create_gripper('panda', root_folder=os.getenv('HOME')+'/cgn')

        for poses in grasps_list:
            gripper_np = gripper_object.get_control_point_tensor(poses.shape[0])
            hom = np.ones((gripper_np.shape[0], gripper_np.shape[1], 1))
            gripper_pts = torch.Tensor(np.concatenate((gripper_np, hom), 2)).transpose(1,2).to(self.device)
            pts = torch.matmul(poses, gripper_pts).transpose(1,2)

            if include_sym:
                sym_gripper_np = gripper_object.get_control_point_tensor(poses.shape[0], symmetric=True)
                sym_gripper_pts = torch.Tensor(np.concatenate((sym_gripper_np, hom), 2)).transpose(1,2).to(self.device)
                sym_pts = torch.matmul(poses, sym_gripper_pts).transpose(1,2)
                pts_list.append([pts, sym_pts])
            else:
                pts_list.append(pts)

        return pts_list
        
    
    def pose_loss(self, pred_grasps, pred_width, pred_successes, labels_dict, gt_dict, sg_i, collide, args):
        '''
        labels_dict
            success (boolean)
            grasps (6d grasps)
        '''
        
        # success_idxs = np.array(labels_dict['init_success_idxs'], dtype=object)
        try:
            success_idxs = list(np.vstack(np.array(labels_dict['success_idxs'], dtype=object))[:,sg_i]) #np.array(labels_dict['success_idxs'], dtype=object)[:,sg_i]
        except:
            print('we got a problem :(')
            from IPython import embed; embed()
        grasp_labels = labels_dict['grasps'][:,sg_i,:]
        width_labels = labels_dict['width'][:,sg_i,:]
        obj_masks = labels_dict['obj_masks']
                    
        # -- GEOMETRIC LOSS --
        # Turn each gripper control point into a pose object
        
        # Find positive grasp labels and corresponding predictions
        grasp_labels = grasp_labels.float().to(self.device) #.view(grasp_labels.shape[0], -1, 4, 4).to(self.device)
        pos_label_list = []
        pos_pred_list = []
        width_label_list = []
        pred_width_list = []
        empty_idx = []
        label_idx_list = []

        obj_mask_list = []

        for batch, idx_list in enumerate(success_idxs):
            if not np.any(idx_list):
                idx_list = np.array([[0, 0]])
            idx_list = idx_list.T
            try:
                point_idxs = idx_list[0].astype(int)
                label_idxs = idx_list[1].astype(int)
            except:
                from IPython import embed; embed()
            obj_mask = np.nonzero(obj_masks[batch])[0]
            obj_mask = np.isin(point_idxs, obj_mask)
            obj_mask_list.append(obj_mask)
            pose_point_idxs = point_idxs #[obj_mask]
            pose_label_idxs = label_idxs #[obj_mask]

            try:
                pos_labels = grasp_labels[batch, pose_label_idxs, :4, :4]
            except:
                print('error')
                from IPython import embed; embed()
            pos_pred = pred_grasps[batch, pose_point_idxs, :4, :4]
            width_labels_masked = width_labels[batch, point_idxs]
            pred_width_masked = pred_width[batch, point_idxs]

            pos_label_list.append(pos_labels)
            pos_pred_list.append(pos_pred)
            width_label_list.append(width_labels_masked)
            pred_width_list.append(pred_width_masked)
            label_idx_list.append(pose_point_idxs)

        # print('width loss')
        # from IPython import embed; embed()
        # -- WIDTH LOSS --
        # calculates loss on 10 grasp width bins using a sigmoid fn and binary cross entropy
        width_loss = torch.tensor(0.0).to(self.device)
        for w_labels, w_pred, c in zip(width_label_list, pred_width_list, collide):
            if not c:
                w_labels = w_labels.to(self.device)
                w_labels = w_labels.view(1, -1)
                w_pred = w_pred.view(1, -1)
                # width_labels = width_labels.reshape(width_labels.shape[0], -1)

                width_loss_fn = nn.MSELoss().to(self.device)
                raw_width_loss = width_loss_fn(w_pred, w_labels)#nn.BCEWithLogitsLoss(pred_width, width_labels)
                width_loss += raw_width_loss
            else:
                pass
        # embed()
        if width_loss != 0:
            width_loss = width_loss/sum(torch.logical_not(collide))

        
        label_pts_list = self.get_key_points(pos_label_list, include_sym=True)
        pred_pts_list = self.get_key_points(pos_pred_list)
        # label_pts_list = []
        # pred_pts_list = []
        # gripper_object = mesh_utils.create_gripper('panda', root_folder=args.root_path)

        # for pos_labels, pos_pred in zip(pos_label_list, pos_pred_list):
        #     gripper_np = gripper_object.get_control_point_tensor(pos_labels.shape[0])
        #     sym_gripper_np = gripper_object.get_control_point_tensor(pos_labels.shape[0], symmetric=True)
        #     hom = np.ones((gripper_np.shape[0], gripper_np.shape[1], 1))
        #     gripper_pts = torch.Tensor(np.concatenate((gripper_np, hom), 2)).transpose(1,2).to(self.device)
        #     sym_gripper_pts = torch.Tensor(np.concatenate((sym_gripper_np, hom), 2)).transpose(1,2).to(self.device)

        #     label_pts = torch.matmul(pos_labels, gripper_pts).transpose(1,2)
        #     sym_label_pts = torch.matmul(pos_labels, sym_gripper_pts).transpose(1,2)
        #     pred_pts = torch.matmul(pos_pred, gripper_pts).transpose(1,2)

        #     label_pts_list.append([label_pts, sym_label_pts])
        #     pred_pts_list.append(pred_pts)

        pred_pts1 = pred_pts_list[0][:,:,:3]
        label_pts1 = label_pts_list[0][0][:,:,:3]
        if args.viz:
            V(pred_pts1.detach().cpu().numpy(), 'pred/', grasps=True)
            V(label_pts1.detach().cpu().numpy(), 'label/', grasps=True)

        s_sig = nn.Sigmoid().to(self.device)
        # -- ADD-S GEOMETRIC LOSS --
        geom_loss_list = [] #torch.Tensor([0]).to(self.device)
        try:
            for success_idx, pred_pts, label_pts, pred_success_list in zip(success_idxs, pred_pts_list, label_pts_list, pred_successes):
                if len(success_idx) != 0:
                    point_success_mask = success_idx[:, 0]
                    pred_pts = pred_pts[:, :, :3]
                    label_pts_1 = label_pts[0][:, :, :3]
                    label_pts_2 = label_pts[1][:, :, :3]

                    pred_success_list = s_sig(pred_success_list)
                    pred_success_masked = pred_success_list[point_success_mask]

                    norm_1 = torch.mean(torch.linalg.norm((pred_pts - label_pts_1), dim=2), dim=1)
                    norm_2 = torch.mean(torch.linalg.norm((pred_pts - label_pts_2), dim=2), dim=1)
                    min_norm = torch.min(norm_1, norm_2)

                    geom_loss = pred_success_masked*min_norm

                    if torch.max(min_norm) > 100:
                        print('geom loss exploded')
                        from IPython import embed; embed()
                    geom_loss_list.append(min_norm)
                else:
                    pass
                    # geom_loss_list.append(torch.tensor([0.0]).to(self.device))
        except Exception as e:
            print(e)
            from IPython import embed; embed()

        # -- APPROACH AND BASELINE LOSSES --
        total_appr_loss = torch.Tensor([0]).to(self.device)
        for pos_labels, pos_pred in zip(pos_label_list, pos_pred_list):
            a_labels = pos_labels[:, :3, 2]
            a_pred = pos_pred[:, :3, 2]
            appr_loss = torch.linalg.norm(a_pred - a_labels, axis=1)
            total_appr_loss += torch.mean(appr_loss)

        return geom_loss_list, width_loss, total_appr_loss, label_idx_list #success_idxs #, approach_loss, baseline_loss

    def goal_loss(self, pred_success, pred_collide, geom_loss, labels_dict, gt_dict, sg_i, args):
        '''
        subgoal collision score loss, per-point confidence loss
        must be called per goal prediction (so that we can do goal forward pass one by one + fit on GPU
        '''
        obj_masks = labels_dict['obj_masks']
        success_labels = np.array(labels_dict['success'])[:,sg_i,:]
        success_labels = success_labels.reshape(success_labels.shape[0], -1) #np.transpose(success_labels, axes=(0,2,1))
        success_labels = torch.Tensor(success_labels).to(self.device)

        noncollide_mask = [True, True, True]
        pred_success = pred_success.view(len(noncollide_mask), -1)
        obj_s_labels = success_labels[noncollide_mask][np.nonzero(obj_masks[noncollide_mask,:,0])]
        obj_s_pred = pred_success[noncollide_mask][np.nonzero(obj_masks[noncollide_mask,:,0])] 

        # -- for visualization --
        pred_s_mask = obj_s_pred 
        
        # -- GRASP CONFIDENCE LOSS --
        # Use binary cross entropy loss on predicted success to find top-k preds with largest loss
        if obj_s_pred.shape[0] < 100:
            obj_k = obj_s_pred.shape[0]
        else:
            obj_k = 100
        conf_loss = torch.topk(self.conf_loss_fn(pred_success[noncollide_mask], success_labels[noncollide_mask]), k=512)[0]
        if len(conf_loss) > 0:
            conf_loss = torch.mean(conf_loss).to(self.device)
            inv_geom_s = []
            for p, l, geom in zip(pred_success, success_labels.type(torch.bool), geom_loss):
                pos_s_loss = self.conf_loss_fn(p[l], torch.ones_like(p[l]))
                inv_geom_s.append(pos_s_loss)
        else:
            print('no conf loss')
            from IPython import embed; embed()
            conf_loss = torch.tensor([0.0]).to(self.device)
            obj_conf_loss = torch.mean(torch.topk(self.conf_loss_fn(obj_s_pred, obj_s_labels), k=obj_k)[0]).to(self.device)

            pos_pred_s = pred_success[labels_dict['success'][:,sg_i,:,0].to(torch.bool)].to(self.device)

            self.pos_weight = torch.tensor([1]).to(self.device)
            conf_sig = nn.Sigmoid().to(self.device)
            pos_loss = torch.mean(self.conf_loss_fn(conf_sig(pos_pred_s), torch.ones_like(pos_pred_s).to(self.device)))*self.pos_weight.to(self.device)
            conf_loss = (pos_loss + conf_loss)/2

        sg_loss = None

        return sg_loss, conf_loss, obj_s_pred, obj_s_labels, sum([torch.mean(i) for i in inv_geom_s])
        
    def SAnet_msg(self, cfg):
        '''
        part of the net that compresses the pointcloud while increasing per-point feature size
        
        cfg: config dict
            radii - nested list of radii for each level
            centers - list of number of neighborhoods to sample for each level
            mlps - list of lists of mlp layers for each level
        '''
        sa_modules = nn.ModuleList()
        input_size = 0
        num_points = 20000
        for r_list, center, mlp_list in zip(cfg['radii'], cfg['centers'], cfg['mlps']):
            layer_modules = []
            feat_cat_size = 0
            for r, mlp_layers in zip(r_list, mlp_list):
                mlp_layers.insert(0, input_size+3)
                module = SAModule((center/num_points), r, MLP(mlp_layers)).to(self.device)
                layer_modules.append(copy.deepcopy(module))
                feat_cat_size += mlp_layers[-1] #keep track of how big the concatenated feature vector is for MSG
            num_points = center
            input_size = feat_cat_size
            self.feat_cat_list.insert(0, feat_cat_size)
            sa_modules.append(nn.ModuleList(copy.deepcopy(layer_modules)))
        return sa_modules

    def SAnet(self, cfg):
        '''
        final module of the set aggregation section
        does not use multi-scale grouping (essentially one MLP applied to the final 128 centers)
        
        cfg: config dict
            mlp - list of mlp layers including input size of 640
        '''
        sa_module = MLP(cfg['mlp'])
        return sa_module

    
    def FPnet(self, cfg):
        '''
        part of net that upsizes the pointcloud

        cfg: config dict
            klist - list of k nearest neighbors to interpolate between
            nnlist - list of unit pointclouds to run between feat prop layers
        '''
        fp_modules = nn.ModuleList()
        input_size = self.feat_cat_list[0]
        for i, layer_list in enumerate(cfg['nnlist']):
            input_size += self.feat_cat_list[i]
            layer_list.insert(0, input_size)
            module = FPModule(3, MLP(layer_list)) # 3 is k in knn interpolate (interpolate from 3 nearest centers)
            input_size = layer_list[-1]
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
        for i, (out_dim, p) in enumerate(zip(cfg['out_dims'], cfg['ps'])):
            in_dim = 128+3
            head = nn.Sequential(nn.Conv1d(in_dim, 128, 1),
                                 nn.BatchNorm1d(128),
                                 nn.Dropout(p),
                                 nn.Conv1d(128, out_dim, 1)).to(self.device)
            head_list.append(head)
        head_list = nn.ModuleList(head_list)
        return head_list
