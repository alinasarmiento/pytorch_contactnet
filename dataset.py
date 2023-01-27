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
from test_meshcat_pcd import viz_scene as VS
import model.utils.mesh_utils as mesh_utils

from IPython import embed

def get_dataloader(data_path, batch, size=None, data_config=None, preloaded=False, args=None): #optim=False, aux=False, viz=False, var=False, just_init=False,
    print('getting dataloader.')
    just_init = False
    viz = args.viz
    if args.model == 'baseline':
        just_init = True
    if not preloaded:
        try:
            demo = args.demo
        except:
            demo = False
        dataset = ContactDataset(data_path, data_config, batch, size=size, viz=viz, just_init=just_init, demo=demo)
    else:
        dataset = SavedDataset(os.path.join(os.getenv('HOME'), 'subgoal-net/preloaded'), data_config, batch, just_init=just_init)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch)
    print('dataloader created.')
    return dataloader

def viz_grasps(grasps, name, freq=100):
    gripper_object = mesh_utils.create_gripper('panda', root_folder='/home/alinasar/subgoal-net/')
    gripper_np = gripper_object.get_control_point_tensor(grasps.shape[0])
    hom = np.ones((gripper_np.shape[0], gripper_np.shape[1], 1))
    gripper_pts = np.concatenate((gripper_np, hom), 2).transpose((0,2,1))
    pts = np.matmul(grasps, gripper_pts).transpose((0,2,1))
    V(pts, name+'/', grasps=True, freq=freq)

class SavedDataset(Dataset):
    def __init__(self, data_path, data_config, batch, just_init):
        self.data_path = data_path
        self.data = os.listdir(self.data_path)
        self.just_init = just_init
        self.data_config = data_config
        self.data_path = os.fsencode(data_path)

    def __getitem__(self, idx):
        data_file = self.data[idx]
        filename = os.path.join(self.data_path, os.fsencode(data_file))
        arrays = load(filename, allow_pickle=True)
        [k0, k1, k2, k3, k4, k5, k6] = arrays.files
        pcd_list, _, target_mask, mean, self.camera_pose, self.gt_contact_info, labels_dict = arrays[k0], arrays[k1], arrays[k2], arrays[k3], arrays[k4], arrays[k5], arrays[k6]

        permute = torch.tensor([0,1]) #torch.randperm(pcd_list.shape[0])
        if self.just_init:
            permute = 0
            
        pcd_list = pcd_list[permute]

        self.gt_contact_info = self.gt_contact_info.item()
        self.gt_contact_info['contact_pts'] = self.gt_contact_info['contact_pts'][permute]
        self.gt_contact_info['grasp_poses'] = self.gt_contact_info['grasp_poses'][permute]
        self.gt_contact_info['base_dirs'] = self.gt_contact_info['base_dirs'][permute]
        self.gt_contact_info['approach_dirs'] = self.gt_contact_info['approach_dirs'][permute]
        self.gt_contact_info['collision_labels'] = np.array(self.gt_contact_info['collision_labels'])[permute]

        labels_dict['success_idxs'] = labels_dict['idxs']
        labels_dict['grasps'] = labels_dict['grasp_poses']
        labels_dict['base_vecs'] = labels_dict['base_dirs']
        labels_dict['appr_vecs'] = labels_dict['approach_dirs']
        
        return pcd_list, permute, target_mask, mean, self.camera_pose, self.gt_contact_info, labels_dict

    def __len__(self):
        return len(self.data)

    
class ContactDataset(Dataset):
    def __init__(self, data_path, data_config, batch, size=None, load_path='/home/alinasar/acronym/test_10', aux=True, viz=False, just_init=False, demo=False):
        self.data = []
        self.optim = False #optim
        self.aux = aux
        self.viz = viz
        self.var = False 
        self.just_init = just_init
        self.data_config = data_config
        self.data_path = os.fsencode(data_path)
        self.load_path = load_path
        self.pcreader = data_utils.PointCloudReader(data_path, batch, raw_num_points=20000, pc_augm_config=data_config['pc_augm'], depth_augm_config=data_config['depth_augm'])
        if size is None:
            self.data = os.listdir(self.data_path)
        else:
            self.data = os.listdir(self.data_path)[:(size+1)]
        self.loaded_size = len(os.listdir(self.load_path))
        self.demo = demo
                        
    def get_contact_info(self, scene):
        gt_contact_info = {}
        if self.aux:
            contact_pts, grasp_poses, base_dirs, approach_dirs, offsets, idcs = data_utils.load_contact_grasps_aux(scene, self.data_config)
            gt_contact_info['collision_labels'] = scene[0]['var_dict'].item()['collision_labels']
        else:
            contact_pts, grasp_poses, base_dirs, approach_dirs, offsets, idcs = data_utils.load_contact_grasps(scene, self.data_config)
        gt_contact_info['contact_pts'] = contact_pts
        gt_contact_info['grasp_poses'] = grasp_poses
        gt_contact_info['base_dirs'] = base_dirs
        gt_contact_info['approach_dirs'] = approach_dirs
        gt_contact_info['offsets'] = offsets
        gt_contact_info['idcs'] = idcs
        return gt_contact_info
    
    def __getitem__(self, idx):
        data_file = self.data[idx]
        print('DATA INDEX', data_file)
        filename = os.path.join(self.data_path, os.fsencode(data_file))
        scene_data = load(filename, allow_pickle=True)
        if not self.demo:
            self.gt_contact_info = self.get_contact_info([scene_data])
            obj_tf = scene_data['init_to_goal']
            obj_paths = scene_data['obj_paths']
            obj_transforms = scene_data['obj_transforms']
            obj_scales = scene_data['obj_scales']
            g = scene_data['goal_tf']

            VS(None, obj_paths, obj_transforms, obj_scales, obj_paths, clear=True)
        else:
            scene_data = scene_data['arr_0'].item()
            self.gt_contact_info= {}
            obj_transforms = scene_data['obj_transforms']            
            for t in obj_transforms:
                t[2,3] += 0.3
            g = scene_data['goal_tf']
            g[2,3] += 0.3
            obj_tf = g @ np.linalg.inv(obj_transforms[scene_data['target_obj']])
            obj_paths = np.array(scene_data['obj_paths']).astype('U66')
            obj_scales = scene_data['obj_scales']
            self.gt_contact_info['goal_tf'] = obj_tf

        # render point clouds
        
        target_obj = scene_data['target_obj']
        if not self.demo:
            target_obj = target_obj.item()
        # obj_tf = scene_data['init_to_goal']

        xr = R.from_euler('x', np.pi, degrees=False)
        x_rot = np.eye(4)
        x_rot[:3, :3] = xr.as_matrix()

        self.pcreader._renderer.change_scene(obj_paths, obj_scales, obj_transforms)
        
        too_small = True
        visible = False
        attempts = 0
        while too_small and attempts < 8:
            # print('finding new scene')
            self.pc_cam, self.camera_pose, self.depth = self.pcreader.render_random_scene(estimate_normals=True, camera_pose=None)
            rot_mat = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            cam2 = rot_mat @ self.camera_pose
            pc_cam2, _, depth2 = self.pcreader.render_random_scene(estimate_normals=True, camera_pose=cam2)            
            segmap, segkeys, seg_pcs = self.pcreader._renderer.render_labels(self.depth, obj_paths, obj_scales, render_pc=True)
            segmap2, segkeys2, seg_pcs2 = self.pcreader._renderer.render_labels(depth2, obj_paths, obj_scales, render_pc=True)
            
            visible = target_obj in seg_pcs.keys()
            visible2 = target_obj in seg_pcs2.keys()
            
            if visible:
                target_pc = seg_pcs[target_obj]    
                init_pc = copy.deepcopy(self.pc_cam)
                target_mask = np.logical_and(np.logical_and(np.isin(init_pc[:,0], target_pc[:,0]), np.isin(init_pc[:,1], target_pc[:,1])), np.isin(init_pc[:,2], target_pc[:,2]))
                target_mask = np.array([target_mask]).T
                too_small = (sum(target_mask) < 1000).item()
            if visible2:
                self.pc_cam = pc_cam2
                self.camera_pose = cam2
                self.depth = depth2
                segmap, segkeys, seg_pcs = segmap2, segkeys2, seg_pcs2
                target_pc = seg_pcs[target_obj]    
                init_pc = copy.deepcopy(self.pc_cam)
                target_mask = np.logical_and(np.logical_and(np.isin(init_pc[:,0], target_pc[:,0]), np.isin(init_pc[:,1], target_pc[:,1])), np.isin(init_pc[:,2], target_pc[:,2]))
                target_mask = np.array([target_mask]).T
                too_small = (sum(target_mask) < 1000).item()
                
            attempts += 1
        if attempts == 8:
            return self.__getitem__(np.random.randint(len(self.data)))

        # transform point cloud to world frame
        pc_hom = np.concatenate((self.pc_cam, np.ones((self.pc_cam.shape[0], 1))), 1).T
        self.init_pc = np.dot(x_rot, pc_hom)
        self.init_pc = np.dot(self.camera_pose, self.init_pc).T

        surrounding_pc = np.invert(target_mask)*self.init_pc
        obj_pc = target_mask*self.init_pc
        
        # transform object pointcloud to goal pose
        init_pose = obj_transforms[target_obj]

        obj_pc = np.matmul(obj_tf, obj_pc.T).T

        self.goal_pc = surrounding_pc + obj_pc
        
        mean = np.mean(self.init_pc[:,:3], axis=0)
        # obj_pc[:,:3] -= mean
        if self.demo:
            v_list = []
        else:
            v_list = scene_data['var_dict'].item()['variations']
        if self.aux:
            # create all variant point clouds
            variant_pcs = []
            for v in v_list:
                o = copy.deepcopy(obj_pc)
                o = np.matmul(v, o.T).T
                var_pc = surrounding_pc + o
                var_pc[:,:3] -= mean
                variant_pcs.append(torch.Tensor(copy.deepcopy(var_pc[:,:3])).float())
        self.init_pc = self.init_pc[:,:3] - mean

        self.goal_pc[:,:3] -= mean

        for ot in obj_transforms:
            ot[:3,3] -= mean
        g[:3,3] -= mean
        if self.demo:
            from test_meshcat_pcd import show_mesh
            show_mesh(None, obj_paths, obj_transforms, obj_scales, ['a','b','c','d','e'], clear=True)
            show_mesh(None, [obj_paths[scene_data['target_obj']]], [g], [obj_scales[scene_data['target_obj']]], ['end'])
        
        scene_dict = copy.deepcopy(dict(scene_data))
        scene_dict.pop('obj_paths')
        if not self.demo:
            scene_dict.pop('raw_paths')
        self.goal_pc = self.goal_pc[:,:3]
        
        pcd_list = torch.stack([torch.Tensor(self.init_pc), torch.Tensor(self.goal_pc), *variant_pcs])

        # dataset debugging
        
        permute = torch.tensor([0,1]) #torch.randperm(pcd_list.shape[0])
        # if self.var:
        #     permute = torch.cat(permute, (torch.randperm(pcd_list.shape[0]-2)+2))
        if self.just_init:
            permute = 0
            
        pcd_list = pcd_list[permute]
        if not self.demo:
            self.gt_contact_info['grasp_poses'] = self.gt_contact_info['grasp_poses'][permute]
            self.gt_contact_info['base_dirs'] = self.gt_contact_info['base_dirs'][permute]
            self.gt_contact_info['approach_dirs'] = self.gt_contact_info['approach_dirs'][permute]
            self.gt_contact_info['contact_pts'] = self.gt_contact_info['contact_pts'][permute]
            self.gt_contact_info['collision_labels'] = np.array(self.gt_contact_info['collision_labels'])[permute]
            if self.var:
                self.gt_contact_info['obj_transforms'] = [obj_transforms]
            self.gt_contact_info['target_obj'] = target_obj
            self.gt_contact_info['goal_tf'] = scene_data['goal_tf']

        if self.optim:
            return torch.Tensor(self.init_pc).float(), torch.Tensor(self.goal_pc).float(), target_mask, mean, self.camera_pose, self.gt_contact_info, scene_dict
        elif self.aux:
            return pcd_list, permute, target_mask, mean, self.camera_pose, self.gt_contact_info
            #return torch.Tensor(self.init_pc).float(), torch.Tensor(self.goal_pc).float(), variant_pcs, target_mask, mean, self.camera_pose, self.gt_contact_info
        else:
            return torch.Tensor(self.init_pc).float(), torch.Tensor(self.goal_pc).float(), target_mask, mean, self.camera_pose, self.gt_contact_info
        
    def __len__(self):
        return len(self.data)
