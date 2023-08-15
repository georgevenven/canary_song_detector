"""
Script Purpose: The point of song detector is to detect songs (CLI)

Inputs from CLI: Weights file, source of folders (folders are already joined)

Outputs: 
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import glob
import shutil
import matplotlib.pyplot as plt
import scipy.io

### Dataloader 

class DetectorDataClass():
    def __init__(self, dir, spec=513):
        self.data = []
        self.spec = spec

        for file in os.listdir(dir):
            self.data.append(os.path.join(dir, file))

    def __getitem__(self, index):
        data = self.data[index]
        mat_data = scipy.io.loadmat(data)
        mat_data = mat_data["song_data"]
        mat_data = mat_data[0][0]

        arr1 = mat_data[0]
        arr2 = mat_data[1]

        # beware if spec shape changes, this might cause error
        if arr1.shape[0] == self.spec:
            spec = torch.Tensor(arr1)
            raw_labels = torch.Tensor(arr2).int()
        else:
            spec = torch.Tensor(arr2)
            raw_labels = torch.Tensor(arr1).int()

        if raw_labels.shape == (0,0):
            song = False
        else:
            song = True

        # labels will be the same length as the song, but it will be filled with 1s between indcies of start and stops
        labels = torch.zeros(size=(spec.shape[1],))
        if song == True:
            num_entries = raw_labels.shape[0]
            for i in range(num_entries):
                if i % 2 == 0:
                    labels[raw_labels[i]:raw_labels[i+1]].fill_(1)
        
        spec = spec.unsqueeze(0)
        
        return spec, labels, song

    def __len__(self):
        return len(self.data) 

###

# copy model #

def song_detector(folders_dir, weights_dir, model):
    # list of that contains lists of (bird_name, song_file_name, endpoints_1 ... endpoints_n)


    model = Model()
    # load model weights 
    state_dict = torch.load(weights_dir)
    model.load_state_dict(state_dict)

    for bird in folders_dir:
        for song in os.path.join(folders_dir, bird):
            # load into  truncated dataset function
            model.forward()

    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Song Detection')
    
    # Add arguments
    parser.add_argument('folders_dir', type=str)
    parser.add_argument('weights_dir', type=str)

    # Parse the arguments
    args = parser.parse_args()
    
    # Use the arguments
    song_detector(args.folders_dir, args.weights_dir, model)
