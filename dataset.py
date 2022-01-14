import os
import sys
import numpy as np
from numpy import load
import data_utils
from torch.utils.data import DataLoader, Dataset

def get_dataloader(data_path, data_config=None):
    dataset = ContactDataset(data_path, data_config)
    dataloader = DataLoader(dataset)
    return dataloader

class ContactDataset(Dataset):
    def __init__(self, data_path, data_config):
        self.data = []
        data_path = os.fsencode(data_path)
        for file in os.listdir(data_path):
            filename = os.fsdecode(file)
            data_dict = {}
            pc_segments = {}
            segmap, rgb, depth, cam_K, pc_full, pc_colors = load_available_input_data(filename, K=None)
            print('Converting depth to point cloud(s)...')
            pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(depth, cam_K, segmap=segmap, rgb=rgb,
                                                                                    skip_border_objects=skip_border_objects, z_range=[0.2, 1.8])
            pc, pc_mean = preprocess_pc_for_inference(pc_full.squeeze(), self._num_input_points, return_mean=True, convert_to_internal_coords=convert_cam_coords)

            data_dict['pcd'] = pc
            data_dict['pcd_mean'] = pc_mean
            data_dict['pcd_segments'] = pc_segments
            data_dict['colors'] = pc_colors

            self.data.append(data_dict)
        print('Data preprocessing complete.')
        
    def __getitem__(self, idx):
        data_dict = self.data[idx]
        pcd = data_dict['pcd']
        pcd_mean = data_dict['pcd_mean']
        pcd_segments = data_dict['pcd_segments']
        pcd_colors = data_dict['colors']

        return pcd, pcd_mean, pcd_segments, pcd_colors

    def __len__(self):
        return len(self.data)
