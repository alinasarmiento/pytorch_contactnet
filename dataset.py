import os
import copy
import sys
import numpy as np
from numpy import load
import data_utils
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.spatial import KDTree, cKDTree
from scipy.spatial.transform import Rotation as R
from test_meshcat_pcd import viz_pcd as V


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
    def __init__(self, data_path, data_config, size=None, overfit_test=False):
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
            self.pc_cam, self.camera_pose, self.depth = self.pcreader.render_random_scene(estimate_normals=True)
                        
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
        if self.multiview:
            if idx%self.max_mv==0:
        '''
        try:
            data_file = self.data[idx]
            filename = '../acronym/scene_contacts/' + os.fsdecode(data_file)
            scene_data = load(filename, allow_pickle=True)
            self.gt_contact_info = self.get_contact_info([scene_data])
        except:
            idx += 1
            return self.__getitem__(idx)
        '''
        if not self.overfit_test:
            data_file = self.data[idx]
            filename = '../acronym/scene_contacts/' + os.fsdecode(data_file)
            scene_data = load(filename, allow_pickle=True)
            self.gt_contact_info = self.get_contact_info([scene_data])
        else:
            scene_data = self.overfit_scene
        '''
        
        # render point clouds
        obj_paths = scene_data['obj_paths']
        for i, path in enumerate(obj_paths):
            fixed_path = '../acronym/models/' + path.split('/')[-1]
            obj_paths[i] = fixed_path
        obj_scales = scene_data['obj_scales']
        obj_transforms = scene_data['obj_transforms']
        
        #if not self.overfit_test:
        self.pcreader._renderer.change_scene(obj_paths, obj_scales, obj_transforms)
        self.pc_cam, self.camera_pose, self.depth = self.pcreader.render_random_scene(estimate_normals=True, camera_pose=None) #self.camera_pose
        #V(self.pc_cam, 'scene/pc_cam')
        
        # transform point cloud to world frame
        pc_hom = np.concatenate((self.pc_cam, np.ones((self.pc_cam.shape[0], 1))), 1).T
        xr = R.from_euler('x', np.pi, degrees=False)
        x_rot = np.eye(4)
        x_rot[:3, :3] = xr.as_matrix()
        self.pc = np.dot(x_rot, pc_hom)
        self.pc = np.dot(self.camera_pose, self.pc).T
        #V(self.pc, 'scene/world__pc')
        
        mean = np.mean(self.pc[:,:3], axis=0)
        self.pc = self.pc[:,:3] - mean
        #V(self.pc, 'scene/norm')

        pcd = self.pc
        #pcd_normals = self.pc_normals[:, :3]

        return torch.Tensor(pcd).float(), mean, self.camera_pose, self.gt_contact_info

    def __len__(self):
        return len(self.data)
