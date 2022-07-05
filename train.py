import os
import os.path
import sys
import argparse
import numpy as np
import math
import time
import traceback

import torch
print(os.getcwd())
print(sys.path)
from model.contactnet import ContactNet
import model.utils.config_utils as config_utils
from data_utils import compute_labels
from dataset import get_dataloader
from torch.utils.tensorboard import SummaryWriter
import copy
from torch_geometric.nn import fps
from test_meshcat_pcd import viz_pcd as V


def initialize_loaders(data_pth, data_config, size=None, include_val=False):
    train_loader = get_dataloader(data_pth, size=size, data_config=data_config)
    if include_val:
        val_loader = get_dataloader(data_pth, data_config)
    else:
        val_loader = None
    return train_loader, val_loader

def initialize_net(config_file, load_model, save_path):
    torch.cuda.empty_cache()
    # Read in config yaml file to create config dictionary
    config_dict = config_utils.load_config(config_file)

    # Init net
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    contactnet = ContactNet(config_dict, device).to(device)
    optimizer = torch.optim.Adam(contactnet.parameters(), lr=0.001) #, weight_decay=0.1)
    if load_model==True:
        print('loading model')
        checkpoint = torch.load(save_path)
        contactnet.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    #for name, param in contactnet.named_parameters(): #state_dict().items():
    #    print(name)
        #print(type(v))
    return contactnet, optimizer, config_dict

def train(model, optimizer, config, train_loader, val_loader=None, epochs=1, save=True, save_pth=None, args=None):
    
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40)
    writer = SummaryWriter()
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(args.epoch_marker,epochs):
        # Train
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(total_norm)
            
            scene_pcds, pcd_mean, cam_poses, gt_dict = data
            pcd_mean = np.expand_dims(pcd_mean, 1)
            print(pcd_mean.shape)
            pcd_mean = torch.Tensor(pcd_mean)
            
            # scene_pcds shape is (batch size, num points, 3)
            data_shape = scene_pcds.shape
            batch_list = torch.arange(0, data_shape[0])
            batch_list = batch_list[:, None].repeat(1, data_shape[1])
            batch_list = batch_list.view(-1).long().to(model.device)

            pcd = scene_pcds.view(-1, data_shape[2]).to(model.device)
            expanded_pcd = copy.deepcopy(pcd.detach().cpu())

            with torch.no_grad():
                idx = fps(expanded_pcd[:, :3], batch_list.detach().cpu(), 2048/20000) # TODO: take out hard coded ratio
                expanded_pcd = expanded_pcd[idx]
                expanded_pcd = expanded_pcd.view(data_shape[0], -1, 3)
                grasp_poses = gt_dict['grasp_poses'] #currently in the wrong shape, need to expand and rebatch for label computation
                grasp_poses = grasp_poses.view(data_shape[0], -1, 4, 4) # B x num_label_points x 4 x 4
                grasp_poses[:,:,:3,3] -= pcd_mean
                # farthest point sample the pointcloud

                gt_points = gt_dict['contact_pts']
                pcd_shape_batched = (gt_points.shape[0], gt_points.shape[2], -1)

                gt_points = gt_points.view(pcd_shape_batched) #.to(model.device)
                gt_points -= pcd_mean
                
                grasp_poses, success_idxs, base_dirs, width, success, approach_dirs = compute_labels(gt_points,
                                                                                        expanded_pcd[:, :, :3],
                                                                                        cam_poses,
                                                                                        gt_dict['base_dirs'],
                                                                                        gt_dict['approach_dirs'],
                                                                                        gt_dict['offsets'],
                                                                                        grasp_poses,
                                                                                        config['data'])

                if args.viz:
                    V(gt_points[0,:,:].cpu().numpy(), 'scene/ground_truth', clear=True)
                    V(pcd[:20000, :3].cpu().numpy(), 'scene/world_pc')
            
                labels_dict = {}
                labels_dict['success_idxs'] = success_idxs
                labels_dict['success'] = success
                labels_dict['grasps'] = grasp_poses
                labels_dict['width'] = width
                labels_dict['base_vecs'] = base_dirs
                labels_dict['appr_vecs'] = approach_dirs

            flag = False
            for idx_list in success_idxs:
                if len(idx_list) == 0:
                    flag = True
                    break
            if flag: continue

            optimizer.zero_grad()
            points, pred_grasps, pred_successes, pred_widths = model(pcd[:, 3:], pos=pcd[:, :3], batch=batch_list.to(model.device), idx=idx.to(model.device), k=None)

            loss_list = model.loss(pred_grasps, pred_successes, pred_widths, labels_dict, args)
            loss = loss_list[-1]
            writer.add_scalar('Loss/total', loss, i)
            writer.add_scalar('Loss/width', loss_list[2], i)
            writer.add_scalar('Loss/conf', loss_list[0], i)
            writer.add_scalar('Loss/add-s', loss_list[1], i)
            writer.add_scalar('Loss/appr', loss_list[3], i)

            loss.backward()
            optimizer.step()
            #running_loss += loss

            # save the model
            current_pth = args.save_path + 'current.pth'
            try:
                f = open(current_pth, 'x')
            except:
                pass
            checkpoint = {'state_dict':model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch':i+1}
            torch.save(checkpoint, current_pth)
            if i%1000 == 0:
                print('[Epoch: %d, Batch: %4d / %4d], Train Loss: %.3f' % (epoch + 1, (i) + 1, len(train_loader), running_loss/10))
                #print('CONF', loss_list[0].item(), 'ADD-S', loss_list[1].item(), 'WIDTH', loss_list[2].item())
                if save:
                    epoch_pth = args.save_path + 'epoch_' + str(epoch+1) + '_i_' + str(i+1) + '.pth'
                    try:
                        f = open(epoch_pth, 'x')
                    except:
                        pass

                    checkpoint = {'state_dict':model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch':i+1}
                    torch.save(checkpoint, epoch_pth)

                running_loss = 0.0

        # Validation
        model.eval()
        if val_loader:
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    scene_pcds, label_dicts  = data
                    points, pred_grasps, pred_successes, pred_widths = model(scene_pcds)
                    val_loss = model.loss(pred_grasps, pred_successes, pred_widths, label_dicts)
            print('Validation Loss: %.3f %%' % val_loss)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to run')
    parser.add_argument('--save_data', type=bool, default=True, help='whether or not to save data (save to path with arg --save_path)')
    parser.add_argument('--config_path', type=str, default='./model/', help='path to config yaml file')
    parser.add_argument('--save_path', type=str, default='sandbox', help='path to save file for main net')
    parser.add_argument('--data_path', type=str, default='/home/alinasar/acronym/scene_contacts', help='path to acronym dataset with Contact-GraspNet folder')
    parser.add_argument('--root_path', type=str, default='/home/alinasar/pytorch_contactnet/', help='root path to repo')
    parser.add_argument('--load_model', type=bool, default=False, help='whether to load saved model, specified with --load_path (otherwise will rewrite)')
    parser.add_argument('--load_path', type=str, default='', help='what path to load the saved model from')
    parser.add_argument('--trainset_size', type=int, default=None, help='size of the train set. if None, will use the entire set (10k cluttered scenes)')
    #parser.add_argument('--i', type=int, default=0, help='sample num to start on')
    parser.add_argument('--uncoupled', type=bool, default=False, help='whether or not to couple the add-s and success multiheads')
    parser.add_argument('--epoch_marker', type=int, default=0, help='what epoch to start on')
    parser.add_argument('--viz', type=bool, default=False, help='live visualization in meshcat')
    args = parser.parse_args()
    print(args)

    if not os.path.isdir('./checkpoints/%s/' %args.save_path):
        os.mkdir('checkpoints/%s/' %args.save_path)
    args.save_path = './checkpoints/%s/' %args.save_path
    
    estop = False
    if (args.save_path != parser.get_default('save_path') and args.load_path=='') or args.load_model==False:
        estop_msg = input('halt! you are about to override any data that is in %s. \n enter y to continue: ' %args.save_path)
        if estop_msg != 'y':
            estop = True
    if not estop:
        contactnet, optimizer, config= initialize_net(args.config_path, args.load_model, args.load_path)
        data_config = config['data']
        train_loader, val_loader = initialize_loaders(args.data_path, data_config, size=args.trainset_size)
        '''
        if args.save_data == True:
            try:
                os.makedirs(args.save_path)
            except:
                pass
        '''
        train(contactnet, optimizer, config, train_loader, val_loader, args.epochs, args.save_data, args.save_path, args)
    else:
        print('exiting')
