import os.path
import pybullet as p
import torch
import sys

sys.path.append('../')
from model.contactnet import ContactNet
import model.utils.config_utils as config_utils
from dataset import ContactDataset
from train import initialize_loaders, initialize_net
import argparse

def infer(model, dataset, config):
    model.eval()
    scene_idx = math.randint(0, len(dataset.data))
    data_file = dataset.data[scene_idx]
    filename = '../acronym/scene_contacts/' + os.fsdecode(data_file)
    scene_data = load(filename)

    obj_paths = scene_data['obj_paths']
    for i, path in enumerate(obj_paths):
        fixed_path = '../acronym/models/' + path.split('/')[-1]
        obj_paths[i] = fixed_path
    obj_scales = scene_data['obj_scales']
    obj_transforms = scene_data['obj_transforms']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, help='path to load model from')
    parser.add_argument('--config_path', type=str, default='./model/', help='path to config yaml file')
    parser.add_argument('--data_path', type=str, default='/home/alinasar/acronym/scene_contacts', help='path to acronym dataset with Contact-GraspNet folder')
    args = parser.parse_args()

    contactnet, config = initialize_net(args.config_path, load_model=True, save_path=args.save_path)
    dataset = ContactDataset(args.data_path, config)
    predicted_grasp = infer(contactnet, dataset, config)
