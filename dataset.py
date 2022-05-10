import os
import sys
import numpy as np
from numpy import load
import data_utils
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.spatial import KDTree, cKDTree
from scipy.spatial.transform import Rotation as R

def get_dataloader(data_path, size=None, data_config=None):
    dataset = ContactDataset(data_path, data_config, size=size, overfit_test=False)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=data_config['batch_size'])
    return dataloader

def crop_pcd(pointcloud, center, save_name, radius=0.5):
    '''
    crops a pointcloud to a sphere of specified radius and center 
    (used to make a little demo pointcloud for overfit testing)
    
    returns: np array (Nx3)
    '''
    knn_tree = cKDTree(pointcloud)
    indices = knn_tree.query_ball_point(center, radius)
    cropped_pcd = pointcloud[indices]
    print('cropping', pointcloud.shape, cropped_pcd.shape)
    np.save(save_name, pointcloud)
    return cropped_pcd, indices

class ContactDataset(Dataset):
    def __init__(self, data_path, data_config, size=None, overfit_test=True):
        self.data = []
        self.data_config = data_config
        data_path = os.fsencode(data_path)
        self.pcreader = data_utils.PointCloudReader(data_path, data_config['batch_size'], pc_augm_config=data_config['pc_augm'], depth_augm_config=data_config['depth_augm'])
        if size is None:
            self.data = os.listdir(data_path)
        else:
            self.data = os.listdir(data_path)[:(size+1)]
        self.overfit_test = overfit_test
        if self.overfit_test:
            data_file = self.data[1]
            filename = '../acronym/scene_contacts/' + os.fsdecode(data_file)
            self.overfit_scene = load(filename)
            self.gt_contact_info = self.get_contact_info([self.overfit_scene])
            self.pc_cam, self.pc_normals, self.camera_pose, self.depth = self.pcreader.render_random_scene(estimate_normals=True)
                        
    def get_contact_info(self, scene):
        contact_pts, grasp_poses, base_dirs, approach_dirs, offsets, idcs = data_utils.load_contact_grasps(scene, self.data_config)
        gt_contact_info = {}
        gt_contact_info['contact_pts'] = contact_pts
        gt_contact_info['grasp_poses'] = grasp_poses
        gt_contact_info['base_dirs'] = base_dirs
        gt_contact_info['approach_dirs'] = approach_dirs
        gt_contact_info['offsets'] = offsets
        gt_contact_info['idcs'] = idcs
        return gt_contact_info
    
    def __getitem__(self, idx):
        # get positive grasp info
        '''
        grasp_transforms = scene_data['grasp_transforms']
        contact_pts = scene_data['scene_cotact_points']
        contacts1, contacts2 = np.split(contact_pts, 2, axis=1) #split contact points into first and second point
        contacts1, contacts2 = contacts1.reshape(-1, 3), contacts2.reshape(-1, 3) 
        offsets = np.linalg.norm(np.subtract(contacts1, contacts2))
        '''
        if not self.overfit_test:
            data_file = self.data[idx]
            filename = '../acronym/scene_contacts/' + os.fsdecode(data_file)
            scene_data = load(filename, allow_pickle=True)
            self.gt_contact_info = self.get_contact_info([scene_data])
        else:
            scene_data = self.overfit_scene
            
        # render point clouds
        obj_paths = scene_data['obj_paths']
        for i, path in enumerate(obj_paths):
            fixed_path = '../acronym/models/' + path.split('/')[-1]
            obj_paths[i] = fixed_path
        obj_scales = scene_data['obj_scales']
        obj_transforms = scene_data['obj_transforms']
        
        #if not self.overfit_test:
        self.pcreader._renderer.change_scene(obj_paths, obj_scales, obj_transforms)
        self.pc_cam, self.pc_normals, self.camera_pose, self.depth = self.pcreader.render_random_scene(estimate_normals=True, camera_pose=None) #self.camera_pose

        '''
        pcd_mean = np.mean(self.pc_cam, axis=0)
        pcd_cam_cent = self.pc_cam - pcd_mean
        pcd_cam_cent_rot = np.matmul(self.camera_pose[:-1, :-1], pcd_cam_cent.T).T
        y_rot_mat = R.from_euler('xyz', [0, np.pi, 0]).as_matrix()
        z_rot_mat = R.from_euler('xyz', [0, 0, np.pi/2]).as_matrix()
        pcd_world = pcd_cam_cent_rot + pcd_mean
        
        # pcd_world = pcd_cam_cent_rot + self.camera_pose[:-1, -1] + pcd_mean
        
        rand_pt = [0, 0, 0]
        # self.pc, _ = crop_pcd(pcd_world, rand_pt, 'obs_pcd')
        self.pc, _ = crop_pcd(pcd_cam_cent_rot, rand_pt, 'obs_pcd')
        # self.pc, _ = crop_pcd(pcd, rand_pt, 'obs_pcd')
        self.gt_contact_info['contact_pts'], crop_idcs = crop_pcd(self.gt_contact_info['contact_pts'][0], rand_pt, 'gt_pcd')
        self.gt_contact_info['grasp_poses'] = self.gt_contact_info['grasp_poses'][:, crop_idcs]
        self.gt_contact_info['base_dirs'] = self.gt_contact_info['base_dirs'][:, crop_idcs]
        self.gt_contact_info['approach_dirs'] = self.gt_contact_info['approach_dirs'][:, crop_idcs]
        self.gt_contact_info['offsets'] = self.gt_contact_info['offsets'][:, crop_idcs]
        self.gt_contact_info['idcs'] = crop_idcs
        '''
        pcd = self.pc_cam[:, :3]
        pcd_normals = self.pc_normals[:, :3]
        
        return torch.Tensor(pcd).float(), torch.Tensor(pcd_normals).float(), self.camera_pose, self.gt_contact_info

    def __len__(self):
        return len(self.data)
