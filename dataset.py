import os
import sys
import numpy as np
import data_utils
from torch.utils.data import DataLoader, Dataset

class ContactDataset(Dataset):
    def __init__(self, data_path, data_config):
        contact_infos = data_utils.load_scene_contacts(data_path, num_test=data_config.test_set_size)
        pos_contact_pts, pos_contact_dirs, pos_contact_approaches, pos_finger_diffs = data_utils.load_contact_grasps(contact_infos, data_config)
        
        
    def __getitem__(self, idx):
        sample = self.data[idx]
        
