import os
import numpy as np
import scipy.signal

import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import os
import csv
import numpy as np
import torch
import gc


import psuedo_label_generation
from data_class import SongDataSet_Image
import matplotlib.pyplot as plt
from model import TweetyBERT
from trainer import ModelTrainer
from inference_functions import replace_short_sequences, pad_and_relabel, plot_spectrogram_and_labels
from analysis import TweetyBERTAnalysis

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


def process_files(src, data_root):
    for i, bird in enumerate(os.listdir(src)):
        bird_path = os.path.join(src, bird)
        if os.path.isdir(bird_path):  # Check if it's a directory
            for day in os.listdir(bird_path):
                day_path = os.path.join(bird_path, day)
                if os.path.isdir(day_path):  # Check if it's a directory
                    for file in tqdm(os.listdir(day_path), desc=f"Processing files for bird {bird} on day {day}"):
                        if file.endswith('.mat'):  # Check if the file is a .mat file
                            mat_to_npy(src=os.path.join(day_path, file), dst=data_root, filename=file)


def inference_on_data_set(kmeans, model, data_root, device, dir, plot_dir="plots", save_plots=True):
    plot_dir = os.path.join(dir, plot_dir)
    os.makedirs(plot_dir, exist_ok=True)  # create directory if it doesn't exist

    # Initialize CSV file
    csv_file_path = os.path.join(dir, "song_segments.csv")
    with open(csv_file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['filename', 'contains_song', 'start_indices', 'stop_indices'])

    #### 1 should always be non-song, and 0 should always be song #### Thats why I will generate an empty spectogram and see the labels and invert them if they are wrong
    test = np.zeros((1, 492, 286))
    test = torch.Tensor(test)
    output_test, _ = model.inference_forward(test.unsqueeze(0).to(device))
    labels_test = kmeans.predict(output_test[0].detach().cpu().numpy())

    if labels_test.all() == 1:
        invert_labels = False 
    else:
        invert_labels = True 

    for i, file in tqdm(enumerate(os.listdir(data_root)), desc="Processing files", total=len(os.listdir(dir))):
        f = np.load(os.path.join(data_root, file), allow_pickle=True)
        spectogram = f['s']
        spectogram = spectogram[8:500,:]
        # Normalize (Z-score normalization)
        mean = spectogram.mean()
        std = spectogram.std()
        spectogram = (spectogram - mean) / (std + 1e-7)  # Adding a small constant to prevent division by zero

        # Replace NaN values with zeros
        spectogram[np.isnan(spectogram)] = 0
        spectogram = torch.from_numpy(spectogram).float().unsqueeze(0)
        output, _ = model.inference_forward(spectogram.unsqueeze(0).to(device))

        labels = kmeans.predict(output[0].detach().cpu().numpy())
        
        if invert_labels:
            # invert labels 
            labels = (1-labels)

        # the order of below functions is important, 1 is NOT SONG 
        labels = replace_short_sequences(labels, target_label=1, min_length=50)
        labels = replace_short_sequences(labels, target_label=0, min_length=200)

        # pad song
        labels = pad_and_relabel(labels, target_label=0, n=50)
        # Save plot if required
        if save_plots:
            plot_spectrogram_and_labels(f['s'], labels, save_dir=plot_dir, file_name=(file.split(".")[0] + ".png"))

        # Detect song sequences
        contains_song = any(label == 0 for label in labels)
        padded_labels = np.pad(labels, (1, 1), 'constant', constant_values=1)  # Pad with non-song labels for easy diff
        diff = np.diff(padded_labels)
        
        # Extract start and stop indices
        start_indices = np.where(diff == -1)[0]  # Transition from 1 to 0 (non-song to song)
        stop_indices = np.where(diff == 1)[0]  # Transition from 0 to 1 (song to non-song)

        # Save to CSV
        with open(csv_file_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([file, contains_song, start_indices.tolist(), stop_indices.tolist()])

if __name__ == "__main__":
    torch.random.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='Song Detection')

    # Add arguments
    parser.add_argument('--bird_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)

    # Parse the arguments
    args = parser.parse_args()

    src = args.bird_dir
    dst = args.output_dir

    data_root = os.path.join(dst, "data")
    train = os.path.join(dst, "train")
    test = os.path.join(dst, "test")

    # Create directories if they don't exist
    for dir_path in [data_root, train, test]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # print("Processing files...")
    # process_files(src=src, data_root=data_root)
    # print("Files processed.")

    # print("Initializing Spectrogram Processor...")
    # processor = psuedo_label_generation.SpectrogramProcessor(data_root=data_root, train_dir=train, test_dir=test, n_clusters=100)
    # print("Spectrogram Processor initialized.")
    
    # print("Clearing train and test directories...")
    # processor.clear_directory(train)
    # processor.clear_directory(test)
    # print("Directories cleared.")

    # print("Generating train and test datasets...")
    # processor.generate_train_test()
    # print("Datasets generated.")

    # print("Generating embeddings...")
    # processor.generate_embedding(samples=5e3, time_bins_per_sample=100, reduction_dims=2)
    # print("Embeddings generated.")
    
    # print("Plotting embeddings and labels...")
    # processor.plot_embedding_and_labels()
    # print("Plots generated.")

    # print("Generating train and test labels...")
    # processor.generate_train_test_labels(reduce=False)
    # print("Labels generated.")

    print("Creating data loaders...")
    train_dataset = SongDataSet_Image(train)
    test_dataset = SongDataSet_Image(test)
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)
    print("Data loaders created.")

    print("Initializing model...")
    model = TweetyBERT(d_transformer=64, nhead_transformer=2, embedding_dim=100, num_labels=100, tau=1, dropout=0.1, dim_feedforward=64, transformer_layers=2, reduced_embedding=27)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.000)
    print("Model initialized.")

    print("Starting training...")
    trainer = ModelTrainer(model, train_loader, test_loader, optimizer, device, max_steps=1001, eval_interval=5e1, save_interval=500)
    trainer.train()
    print("Training complete.")

    print("Starting analysis...")
    analysis = TweetyBERTAnalysis(train_loader, model, device)
    
    print("Fitting k-means...")
    kmeans = analysis.fit_kmeans()
    print("k-means fitted.")

    print("Collecting data...")
    analysis.collect_data()
    print("Data collected.")

    print("Plotting UMAP...")
    analysis.plot_umap(output_dir=dst)
    print("UMAP plotted.")

    inference_on_data_set(kmeans=kmeans, model=model, device=device, dir=dst, plot_dir="plots", save_plots=True, data_root=data_root)

    print("done!")







    

