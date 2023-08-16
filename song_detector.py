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

# Dataloader Function
def load_song(self, file_path):
    mat_data = scipy.io.loadmat(file_path)
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
    
    # amplitude 

    return spec

class Song_Detector_Model(nn.Module):
    def __init__(self, d_transformer, nhead_transformer, embedding_dim, num_labels, tau=0.1, dropout=0.1, transformer_layers=1,dim_feedforward=64):
        super(aviaBERT, self).__init__()
        self.tau = tau
        self.num_labels = num_labels
        self.dropout = dropout

        # TweetyNet Front End
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=(8, 1), stride=(8, 1))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=(8, 1), stride=(8, 1))

        # Positional Encoding
        self.pos_conv1 = nn.Conv1d(d_transformer, d_transformer, kernel_size=3, padding=1, dilation=1)
        self.pos_conv2 = nn.Conv1d(d_transformer, d_transformer, kernel_size=3, padding=2, dilation=2)

        # transformer
        self.transformerProjection = nn.Linear(512, d_transformer)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_transformer, nhead=nhead_transformer, batch_first=True, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=transformer_layers)
        self.transformerDeProjection = nn.Linear(d_transformer, embedding_dim)
        self.onedconv = nn.Conv1d(embedding_dim, 1, kernel_size=31, stride=1, padding=15)

    def convolutional_positional_encoding(self, x):
        pos = F.relu(self.pos_conv1(x))
        pos = F.relu(self.pos_conv2(pos))
        return pos

    def feature_extractor_forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.flatten(1,2)
        return x

    def transformer_forward(self, x):
        # project the input to the transformer dimension
        x = x.permute(0,2,1)
        x = self.transformerProjection(x)
        x = x.permute(0,2,1)

        # add convolutional positional encoding
        pos_enc = self.convolutional_positional_encoding(x)
        x = x + pos_enc
        x = x.permute(0,2,1)
        x = self.transformer_encoder(x)
        x = self.transformerDeProjection(x)
        return x

    def forward(self, x):
        x = self.feature_extractor_forward(x)
        x = self.transformer_forward(x)
        x = x.permute(0,2,1)
        x = self.onedconv(x)
        x = x.permute(0,2,1)
        return x

    def BCE_loss(self, y_pred, y_true, weights, too_small=50, scaling_factor=.00):
        # Convert y_pred to binary tensor for > 0.5 and < 0.5
        binary_pred_over_half = (y_pred > 0.5).float()
        binary_pred_under_half = (y_pred < 0.5).float()

        # Define a kernel of size too_small with all ones.
        kernel = torch.ones(too_small).to(y_pred.device)

        # Perform 1D convolution for > 0.5.
        conv_result_over_half = F.conv1d(binary_pred_over_half.unsqueeze(1), kernel.view(1,1,-1), padding=too_small-1).squeeze(1)
        
        # Perform 1D convolution for < 0.5.
        conv_result_under_half = F.conv1d(binary_pred_under_half.unsqueeze(1), kernel.view(1,1,-1), padding=too_small-1).squeeze(1)

        # A mask for regions of > 0.5 that are not continuous.
        non_continuous_mask_over_half = (conv_result_over_half < too_small) & (conv_result_over_half > 0)

        # A mask for regions of < 0.5 that are not continuous.
        non_continuous_mask_under_half = (conv_result_under_half < too_small) & (conv_result_under_half > 0)

        # Find the actual indices for > 0.5.
        non_continuous_indices_over_half = torch.nonzero(non_continuous_mask_over_half).squeeze()
        non_continuous_indices_over_half = non_continuous_indices_over_half.flatten()
        count_non_continuous_over_half = (non_continuous_indices_over_half > -1).sum().item()
        
        # Find the actual indices for < 0.5.
        non_continuous_indices_under_half = torch.nonzero(non_continuous_mask_under_half).squeeze()
        non_continuous_indices_under_half = non_continuous_indices_under_half.flatten()
        count_non_continuous_under_half = (non_continuous_indices_under_half > -1).sum().item()

        # Expand weights tensor to have the same shape as y_pred
        weights = weights.unsqueeze(dim=1)
        weights = weights.expand(weights.shape[0], y_pred.shape[1]).to(y_pred.device)

        loss_fn = torch.nn.BCEWithLogitsLoss(weight=weights)
        loss = loss_fn(input = y_pred, target = y_true)
        
        # Combine the penalties for regions over and under 0.5
        total_penalty = count_non_continuous_over_half + count_non_continuous_under_half

        return loss * (1 + total_penalty * scaling_factor)

def song_detector(folders_dir, weights_dir, model):
    # list of that contains lists of (bird_name, song_file_name, endpoints_1 ... endpoints_n)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Song_Detector_Model().to(device)
    state_dict = torch.load(weights_dir)
    model.load_state_dict(state_dict)

    for bird in folders_dir:
        for song in os.path.join(folders_dir, bird):
            song_path = os.path.join(folders_dir, bird, song)
            song = load_song(song_path)
            song.to(device)
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
