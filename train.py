import os
import os.path
import sys
import argparse
import numpy as np
import math
import time
import traceback

import torch
from model.contactnet_kp import ContactNet as cnkp
import model.utils.config_utils as config_utils
from data_utils import compute_labels, compute_labels_aux
from dataset import get_dataloader
from torch.utils.tensorboard import SummaryWriter
import copy
from torch_geometric.nn import fps
from test_meshcat_pcd import viz_pcd as V
from pytictoc import TicToc

def initialize_loaders(data_pth, data_config, batch, size=None, include_val=False, val_path=None, preloaded=False, args=None):
    print('initializing loaders')
    if batch==None:
        batch = data_config['batch_size']
    train_loader = get_dataloader(data_pth, batch, size=size, data_config=data_config, preloaded=preloaded, args=args)
    if include_val:
        val_loader = get_dataloader(val_path, batch, size=8, data_config=data_config, args=args)
    else:
        val_loader = None
    return train_loader, val_loader

def initialize_net(config_file, load_model, save_path, args):
    print('initializing net')
    torch.cuda.empty_cache()

    # Read in config yaml file to create config dictionary
    config_dict = config_utils.load_config(config_file)

    # Init net
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = cnkp(config_dict, device, args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #, weight_decay=0.1)
    if load_model==True:
        print('loading model')
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer, config_dict

def train(model, optimizer, config, train_loader, val_loader=None, epochs=1, save=True, save_pth=None, args=None):

    t = TicToc()
    
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40)
    writer = SummaryWriter()
    torch.autograd.set_detect_anomaly(True)
    inv_weight_mult = 0
    if args.inv_weight:
        inv_weight_mult = 1
    for epoch in range(args.epoch_marker,epochs):
        # Train
        model.train()
        running_loss = 0.0
        t.tic()
        for i, data in enumerate(train_loader):
            if args.preloaded:
                pcd_list, permute, obj_masks, pcd_mean, cam_poses, gt_dict, labels_dict = data
            pcd_list, permute, obj_masks, pcd_mean, cam_poses, gt_dict = data
            pcd_mean = np.expand_dims(pcd_mean, 1)
            pcd_mean = torch.Tensor(pcd_mean)
            t.toc('Pointclouds retrieved in', restart=True)
            # scene_pcds shape is (batch size, num points, 3)

            # data_shape = pcd_list[0].shape
            b = pcd_list.shape[0]

            init_scene_pcds = pcd_list[permute==0]
            data_shape = init_scene_pcds.shape
            batch_list = torch.arange(0, data_shape[0])
            batch_list = batch_list[:, None].repeat(1, data_shape[1])
            batch_list = batch_list.view(-1).long().to(model.device)

            init_pcd = init_scene_pcds.view(-1, data_shape[2]).to(model.device)
            # end_pcd = goal_scene_pcds.view(-1, data_shape[2]).to(model.device)
            if args.viz:
                V(init_pcd.detach().cpu().numpy(), 'init', clear=True)
                # V(end_pcd.detach().cpu().numpy(), 'end')

            i_pcd = copy.deepcopy(init_pcd).detach().cpu()
            with torch.no_grad():
                idx = fps(i_pcd[:, :3], batch_list.detach().cpu(), 2048/20000) # TODO: take out hard coded ratio
                pcd_lc = torch.flatten(pcd_list.transpose(1,2), start_dim=0, end_dim=1).transpose(0,1)[:,idx,:] # pointcloud for label compute (LC)
                i_pcd = i_pcd[idx]                
                i_pcd = i_pcd.view(data_shape[0], -1, 3)
                pcd_lc = pcd_lc.view(pcd_lc.shape[0], b, -1, 3).transpose(0,1).detach().cpu()

                obj_masks = obj_masks.view(-1, 1)
                obj_masks = obj_masks[idx]
                obj_masks = obj_masks.view(data_shape[0], -1, 1)
                
                gt_dict['grasp_poses'][:,:,:,:3,3] -= pcd_mean.unsqueeze(1)
                gt_dict['contact_pts'] -= pcd_mean.unsqueeze(1)

                if args.model == 'kp' or args.model=='cgn':
                    grasp_poses, success_idxs, base_dirs, approach_dirs, width, success, collision_labels = compute_labels_aux(gt_dict, pcd_lc,
                                                                                                                               cam_poses, config['data'])            
                else:
                    grasp_poses, success_idxs, base_dirs, approach_dirs, width, success, collision_labels = compute_labels(gt_dict, pcd_lc,
                                                                                                                           cam_poses, config['data'])            

                labels_dict = {}
                labels_dict['success_idxs'] = success_idxs
                labels_dict['success'] = success
                labels_dict['grasps'] = grasp_poses
                labels_dict['width'] = width
                labels_dict['base_vecs'] = base_dirs
                labels_dict['appr_vecs'] = approach_dirs
                labels_dict['obj_masks'] = obj_masks.detach().cpu().numpy()

            t.toc('Computed labels in', restart=True)
            
            # pcd_list = [init_pcd, end_pcd, *[v.view(-1, 3).to(model.device) for v in variant_pcs]]
            pcd_list = torch.flatten(pcd_list.transpose(1,2), start_dim=0, end_dim=1).transpose(0,1).to(model.device)
            
            for sg_i, (pcd, collide) in enumerate(zip(pcd_list, collision_labels.T)):

                points, pred_grasps, pred_successes, pred_widths, _,  pred_collide = model(pcd[:, 3:], pcd[:, :3], batch_list.to(model.device), idx.to(model.device), obj_masks)
                t.toc('Forward pass '+str(sg_i)+' took',restart=True)
                geom_loss, width_loss, appr_loss, success_idx_scene = model.pose_loss(pred_grasps, pred_widths, pred_successes, labels_dict, gt_dict, sg_i, collide, args)
                t.toc('pose loss computation took', restart=True)
                if args.model == 'kp':
                    sg_loss = model.goal_loss_sg(pred_successes, pred_collide, geom_loss, labels_dict, gt_dict, sg_i, args)
                else:
                    sg_loss = model.goal_loss(pred_successes, pred_collide, geom_loss, labels_dict, gt_dict, sg_i, args)
                t.toc('success loss computation took', restart=True)
                inv_geom_loss = sg_loss[-1]
                
                if args.viz_s:
                    # pred_s_mask = sg_loss[2] > 0.2
                    label_s_mask = sg_loss[3]
                    print('s viz')
                    # from IPython import embed; embed()
                    g = grasp_poses.view(-1,4,4)[idx]
                    obj_grasps = g[obj_masks[0,:,0]]
                    # pred_o = obj_grasps[pred_s_mask]
                    label_o = obj_grasps[label_s_mask.bool()]

                    from dataset import viz_grasps
                    V(pcd.detach().cpu().numpy(), 'pcd', clear=True)
                    # viz_grasps(pred_o.detach().cpu().numpy(), 'predicted', freq=1)
                    viz_grasps(label_o.detach().cpu().numpy(), 'label', freq=1)
                    
                    from IPython import embed; embed()

                sg_score_loss  = sg_loss[0]
                s_loss = sg_loss[1]

                if len(geom_loss) > 0:
                    geom_loss = torch.mean(torch.cat(geom_loss))
                else:
                    geom_loss = torch.tensor([0.0]).to(model.device)
                
                loss = model.config['loss']['conf_mult']*s_loss  + model.config['loss']['add_s_mult']*geom_loss + model.config['loss']['width_mult']*width_loss
                if loss > 100:
                    print('huge loss!')
                    # from IPython import embed; embed()
                print('loss:', loss.item())
                t.tic()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()
                t.toc('backward pass took')


            # Parameter inspection (debugging)
            params = 0.0
            for param in model.parameters():
                params += torch.sum(torch.abs(param.clone().flatten().detach().cpu()))

            num = epoch*len(train_loader) + i
            writer.add_scalar('Debug/param_sum', params.item(),num)
            writer.add_scalar('Loss/total', loss.item(), num)
            writer.add_scalar('Loss/width', width_loss.item(), num)
            writer.add_scalar('Loss/conf', s_loss.item(), num)
            writer.add_scalar('Loss/add-s', geom_loss.item(), num)
            # writer.add_scalar('Loss/sg_score', sg_score_loss.item(), num)
            writer.add_scalar('Loss/appr', appr_loss.item(), num)
            writer.add_scalar('Loss/inv_geom', inv_geom_loss.item(), num)

            # save the model
            current_pth = args.save_path + 'current.pth'
            try:
                f = open(current_pth, 'x')
            except:
                pass
            if save:
                checkpoint = {'state_dict':model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch':i+1}
                torch.save(checkpoint, current_pth)
            if epoch%10 == 0:
                if save:
                    print('saving checkpoint')
                    epoch_pth = args.save_path + 'iteration_' + str(num)
                    checkpoint = {'state_dict':model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch':i+1}
                    torch.save(checkpoint, epoch_pth)                


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to run')
    parser.add_argument('--save_data', type=bool, default=True, help='whether or not to save data (save to path with arg --save_path)')
    parser.add_argument('--config_path', type=str, default='./model/', help='path to config yaml file')
    parser.add_argument('--save_path', type=str, default='sandbox', help='path to save file for main net')
    parser.add_argument('--data_path', type=str, default='/home/alinasar/acronym/test_scenes', help='path to acronym dataset with Contact-GraspNet folder')
    parser.add_argument('--val_path', type=str, default='/home/alinasar/acronym/validation', help='path to acronym dataset with Contact-GraspNet folder')
    parser.add_argument('--root_path', type=str, default='/home/alinasar/subgoal-net/', help='root path to repo')
    parser.add_argument('--load_model', type=bool, default=False, help='whether to load saved model, specified with --load_path (otherwise will rewrite)')
    parser.add_argument('--load_path', type=str, default='', help='what path to load the saved model from')
    parser.add_argument('--trainset_size', type=int, default=None, help='size of the train set. if None, will use the entire set (10k cluttered scenes)')
    parser.add_argument('--epoch_marker', type=int, default=0, help='what epoch to start on')
    parser.add_argument('--object_only', type=bool, default=False, help='whether to train on all positives or just target-object positives')
    parser.add_argument('--viz', type=bool, default=False, help='live visualization in meshcat')
    parser.add_argument('--batch', type=int, default=None, help='optional batch size. defaults to config file')
    parser.add_argument('--model', type=str, default='kp', help='model architecture to load [kp, baseline, cgn]. defaults to kp.')
    parser.add_argument('--inv_weight', type=bool, default=True)
    parser.add_argument('--obj_s', type=int, default=0, help='whether or not to add extra weighted success term focused on object')
    parser.add_argument('--viz_s', type=bool, default=False)
    parser.add_argument('--preload', type=bool, default=False)
    parser.add_argument('--pos_weight', default=None, help='weight of positive points in conf loss (default None)')
    args = parser.parse_args()
    print(args)

    if not os.path.isdir('./checkpoints/%s/' %args.save_path):
        os.mkdir('checkpoints/%s/' %args.save_path)
    args.save_path = './checkpoints/%s/' %args.save_path

    contactnet, optimizer, config= initialize_net(args.config_path, args.load_model, args.load_path, args=args)
    data_config = config['data']
    train_loader, val_loader = initialize_loaders(args.data_path, data_config, args.batch, include_val=False, val_path=args.val_path, preloaded=args.preload, size=args.trainset_size, args=args)
    train(contactnet, optimizer, config, train_loader, val_loader, args.epochs, args.save_data, args.save_path, args)
