import torch
import torchaudio
import torchaudio.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import librosa
import soundfile as sf
import shutil
from sklearn.cluster import KMeans
from tqdm import tqdm
import seaborn as sns
from sklearn import cluster
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import argparse

import psuedo_label_generation
from data_class import SongDataSet_Image
import matplotlib.pyplot as plt
from model import TweetyBERT
from trainer import ModelTrainer

#############################################################################################################
    

def mat_to_npy(src, dst, filename, spec_height=513):
    """
    Params:
    - file dir
    - 
    """
    mat_data = scipy.io.loadmat(src)
    mat_data = mat_data["song_data"]
    mat_data = mat_data[0][0]

    arr1 = mat_data[0]
    arr2 = mat_data[1]

    # beware if spec shape changes, this might cause error
    if arr1.shape[0] == spec_height:
        spec = arr1
        raw_labels = arr2.astype(int)
    else:
        spec = arr2
        raw_labels = arr1.astype(int)

    if raw_labels.shape == (0,0):
        song = False
    else:
        song = True

    # labels will be the same length as the song, but it will be filled with 1s between indcies of start and stops
    labels = np.zeros((spec.shape[1],))
    if song:
        num_entries = raw_labels.shape[0]
        for i in range(num_entries):
            if i % 2 == 0:
                start = raw_labels[i].item()
                stop = raw_labels[i + 1].item()
                labels[start:stop] = 1  # Using NumPy's slicing
    
    # Save spec, labels, and song as .npz file

    labels = labels.reshape(1, -1)
    save_name = os.path.join(dst, filename)
    np.savez(save_name, s=spec, labels=labels, song=song)

## May have to re extend back to multiple birds 
def process_files(src, data_root):
    for i, bird in enumerate(src):
        for day in os.listdir(os.path.join(src, bird)): 
            d = os.path.join(src, bird, day)
            for file in tqdm(os.listdir(d), desc=f"Processing files for bird {bird} on day {day}"):
                npy_file = mat_to_npy(src = os.path.join(src, bird, day, file), dst= data_root, filename= file)
        


if __name__ == "__main__":
    torch.random.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='Song Detection')
    
    # Add arguments
    parser.add_argument('bird_dir', type=str)
    parser.add_argument('output_dir', type=str)

    # Parse the arguments
    args = parser.parse_args()
    
    src = args.bird_dir
    dst = args.output_dir

    data_root = os.path.join(dst, "/data")
    train = os.path.join(dst, "/train")
    test = os.path.join(dst, "/test")

    process_files(src=src, data_root=data_root)

    processor = psuedo_label_generation.SpectrogramProcessor(data_root=data_root, train_dir=train, test_dir=test, n_clusters=100)

    ### CAREFUL
    processor.clear_directory(train)
    processor.clear_directory(test)
    ### CAREFUL 

    processor.generate_train_test()
    # this is 5,000 samples * num times in samples (10 in the case of 100 timebins in 1000 timebin segment)
    processor.generate_embedding(samples=5e3, time_bins_per_sample=100, reduction_dims=2)
    processor.plot_embedding_and_labels()
    processor.generate_train_test_labels(reduce=False)


    train_dataset = SongDataSet_Image(train)
    test_dataset = SongDataSet_Image(test)
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

    model = TweetyBERT(d_transformer=64, nhead_transformer=2, embedding_dim=100, num_labels=100, tau=1, dropout=0.1, dim_feedforward=64, transformer_layers=2, reduced_embedding= 27)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.000)

    # Usage:
    trainer = ModelTrainer(model, train_loader, test_loader, optimizer, device, max_steps=1001, eval_interval=5e1, save_interval=500)
    trainer.train()

    from analysis import TweetyBERTAnalysis

    analysis = TweetyBERTAnalysis(train_loader, model, device)

    # Fit k-means to initialize the kmeans attribute
    analysis.fit_kmeans()
    # Now, collect the data
    analysis.collect_data()
    # Calculate the F1 score
    f1 = analysis.calculate_f1_score()
    print(f'F1 Score: {f1}')



    

