import os
import sys
import numpy as np
from numpy import load
import data_utils
from torch.utils.data import DataLoader, Dataset

def get_dataloader(data_path, data_config=None):
    #data_path = os.path.join(os.getcwd(), data_path)
    dataset = ContactDataset(data_path, data_config)
    dataloader = DataLoader(dataset, batch_size=data_config['batch_size'])
    return dataloader

def extract_point_clouds(self, depth, K, segmap=None, rgb=None, z_range=[0.2,1.8], segmap_id=0, skip_border_objects=False, margin_px=5):
    """
    Converts depth map + intrinsics to point cloud. 
    If segmap is given, also returns segmented point clouds. If rgb is given, also returns pc_colors.
    Arguments:
        depth {np.ndarray} -- HxW depth map in m
        K {np.ndarray} -- 3x3 camera Matrix
    Keyword Arguments:
        segmap {np.ndarray} -- HxW integer array that describes segeents (default: {None})
        rgb {np.ndarray} -- HxW rgb image (default: {None})
        z_range {list} -- Clip point cloud at minimum/maximum z distance (default: {[0.2,1.8]})
        segmap_id {int} -- Only return point cloud segment for the defined id (default: {0})
        skip_border_objects {bool} -- Skip segments that are at the border of the depth map to avoid artificial edges (default: {False})
        margin_px {int} -- Pixel margin of skip_border_objects (default: {5})
    Returns:
        [np.ndarray, dict[int:np.ndarray], np.ndarray] -- Full point cloud, point cloud segments, point cloud colors
    """
    if K is None:
        raise ValueError('K is required either as argument --K or from the input numpy file')
            
    # Convert to pc 
    pc_full, pc_colors = depth2pc(depth, K, rgb)

    # Threshold distance
    if pc_colors is not None:
        pc_colors = pc_colors[(pc_full[:,2] < z_range[1]) & (pc_full[:,2] > z_range[0])] 
    pc_full = pc_full[(pc_full[:,2] < z_range[1]) & (pc_full[:,2] > z_range[0])]
        
    # Extract instance point clouds from segmap and depth map
    pc_segments = {}
    if segmap is not None:
        pc_segments = {}
        obj_instances = [segmap_id] if segmap_id else np.unique(segmap[segmap>0])
        for i in obj_instances:
            if skip_border_objects and not i==segmap_id:
                obj_i_y, obj_i_x = np.where(segmap==i)
                if np.any(obj_i_x < margin_px) or np.any(obj_i_x > segmap.shape[1]-margin_px) or np.any(obj_i_y < margin_px) or np.any(obj_i_y > segmap.shape[0]-margin_px):
                    print('object {} not entirely in image bounds, skipping'.format(i))
                    continue
            inst_mask = segmap==i
            pc_segment,_ = depth2pc(depth*inst_mask, K)
            pc_segments[i] = pc_segment[(pc_segment[:,2] < z_range[1]) & (pc_segment[:,2] > z_range[0])] #regularize_pc_point_count(pc_segment, grasp_estimator._contact_grasp_cfg['DATA']['num_point'])

    return pc_full, pc_segments, pc_colors

class ContactDataset(Dataset):
    def __init__(self, data_path, data_config):
        self.data = []
        data_path = os.fsencode(data_path)
        self.pcreader = data_utils.PointCloudReader(data_path, data_config['batch_size'], pc_augm_config=data_config['pc_augm'], depth_augm_config=data_config['depth_augm'])
        self.data = os.listdir(data_path)

        '''
        for file in os.listdir(data_path):
            filename = os.fsdecode(file)
            data_dict = {}
            pc_segments = {}
            pc_cam, pc_normals, camera_pose, depth, cam_mat = pcreader.render_random_scene(estimate_normals=True)

            #segmap, rgb, depth, cam_K, pc_full, pc_colors = data_utils.load_available_input_data(filename, K=None)
            #print('Converting depth to point cloud(s)...')
            #pc_full, pc_segments, pc_colors = extract_point_clouds(depth, cam_mat, segmap=segmap, rgb=rgb,
            #                                                                        skip_border_objects=skip_border_objects, z_range=[0.2, 1.8])
            #pc, pc_mean = preprocess_pc_for_inference(pc_full.squeeze(), self._num_input_points, return_mean=True, convert_to_internal_coords=convert_cam_coords)

            data_dict['pcd'] = pc
            data_dict['pcd_mean'] = pc_mean
            data_dict['pcd_segments'] = pc_segments
            data_dict['colors'] = pc_colors

            self.data.append(data_dict)
        print('Data preprocessing complete.')
        '''

    def __getitem__(self, idx):
        data_file = self.data[idx]
        #fixed_name = os.fsdecode(data_file).split('/')[-1]
        #print(fixed_name)
        filename = '../acronym/scene_contacts/' + os.fsdecode(data_file)
        scene_data = load(filename)
        obj_paths = scene_data['obj_paths']
        for i, path in enumerate(obj_paths):
            fixed_path = '../acronym/models/' + path.split('/')[-1]
            obj_paths[i] = fixed_path
        #print(obj_paths)
        obj_scales = scene_data['obj_scales']
        obj_transforms = scene_data['obj_transforms']

        self.pcreader._renderer.change_scene(obj_paths, obj_scales, obj_transforms)
        pc_cam, pc_normals, camera_pose, depth, cam_mat = self.pcreader.render_random_scene(estimate_normals=True)
        
        pcd = pc_cam[:, :3]
        pcd_normals = pc_normals[:, :3]
        
        return pcd, pcd_normals, camera_pose

    def __len__(self):
        return len(self.data)
