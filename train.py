import os
import os.path
import sys
import argparse
import numpy as np
import math

#TO-DO: import data and dataloader
import torch
from contactnet import ContactNet
import config
import utils.config_utils as config_utils
from dataloader import ContactDataset

def initialize_loaders(data_pth, include_val=False):
    # TO-DO

def initialize_net(config_file):
    # Read in config yaml file to create config dictionary
    config_dict = config_utils.load_config(config_file)
    
    # Init net
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    contactnet = ContactNet(config_dict).to(device)
    return contactnet, config_dict

def train(model, config, train_loader, val_loader=None, epochs=1, save=True, save_pth=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    for epoch in range(epochs):
        # Train
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            scene_pcds, label_dicts = data
            optimizer.zero_grad()

            pred_grasps, pred_successes, pred_widths = model(scene_pcds, k=None)
            loss = model.loss(pred_grasps, pred_successes, pred_widths, label_dicts)

            loss.backward(retain_graph=True)
            optimizer.step()
            running_loss += loss
            
            if i%10 == 9:
                print('[Epoch: %d, Batch: %4d / %4d], Train Loss: %.3f' % (epoch + 1, (i) + 1, len(train_loader), running_loss/10))
                running_loss = 0.0

        # Validation
        model.eval()
        if val_loader:
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    scene_pcds, label_dicts  = data
                    pred_grasps, pred_successes, pred_widths = model(scene_pcds)
                    val_loss = model.loss(pred_grasps, pred_successes, pred_widths, label_dicts)
            print('Validation Loss: %.3f %%' % val_loss)

        # save the model
        if save:
            torch.save(model.state_dict(), save_pth)

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to run')
    parser.add_argument('--save_data', type=bool, default=True, help='whether or not to save data (save to path with arg --save_path)')
    parser.add_argument('--config_path', type=str, default='./config.yaml', help='path to config yaml file')
    parser.add_argument('--save_path', type=str, default='../data/model_save.pth', help='path to save file for main net')
    parser.add_argument('--data_path', type=str, default='../../acronym', help='path to acronym dataset with Contact-GraspNet folder')
    args = parser.parse_args()

    # initialize dataloaders
    train_loader, val_loader = initialize_loaders(args.data_path, True)
    
    contactnet, config = initialize_net(args.config_path)
    train(contactnet, config, train_loader, val_loader, args.epochs, args.save_data, args.save_path)
