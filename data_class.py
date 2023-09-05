import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt

class SongDataSet_Image(Dataset):
    def __init__(self, file_dir, num_classes=100, threshold=5):
        self.file_path = []
        self.num_classes = num_classes
        self.threshold = threshold  # amplitude threshold

        for file in os.listdir(file_dir):
            self.file_path.append(os.path.join(file_dir, file))

    def __getitem__(self, index):
        file_path = self.file_path[index]

        data = np.load(file_path)
        spectogram = data['s']
        ground_truth_labels = data['labels']
        psuedo_labels = data['new_labels']

        # Normalize (Z-score normalization)
        mean = spectogram.mean()
        std = spectogram.std()
        spectogram = (spectogram - mean) / (std + 1e-7)  # Adding a small constant to prevent division by zero

        # Replace NaN values with zeros
        spectogram[np.isnan(spectogram)] = 0

        # Convert label and psuedo label to one-hot encoding
        ground_truth_labels = torch.tensor(ground_truth_labels, dtype=torch.int64)
        ground_truth_labels = F.one_hot(ground_truth_labels, num_classes=self.num_classes).float()
        
        psuedo_labels = torch.tensor(psuedo_labels, dtype=torch.int64)
        psuedo_labels = F.one_hot(psuedo_labels, num_classes=self.num_classes).float()
        psuedo_labels = torch.squeeze(psuedo_labels, dim=0)
        ground_truth_labels = torch.squeeze(ground_truth_labels, dim=0)

        # Convert to torch tensors
        spectogram = torch.from_numpy(spectogram).float().unsqueeze(0)

        filename = os.path.basename(file_path)

        return spectogram, psuedo_labels, ground_truth_labels

    def __len__(self):
        return len(self.file_path)

def plot_spectrogram_and_labels(spec, labels):
    labels = torch.argmax(labels, dim=-1).numpy()  # Converting to numpy for easier indexing

    fig, ax1 = plt.subplots(figsize=(20, 4))

    # Plot the spectrogram on the first axis
    img = ax1.imshow(spec, aspect='auto', origin='lower', cmap='inferno')
    ax1.set_xlabel('Time frames')
    ax1.set_ylabel('Frequency bins')
    ax1.set_title('Spectrogram and Ground Truth Labels')

    # Adding color bar for reference
    cbar = plt.colorbar(img, ax=ax1)
    cbar.set_label('Amplitude', rotation=90)

    # Find out the maximum frequency bin index to position the line at the top
    max_frequency_index = spec.shape[0] - 1

    # Plotting small horizontal lines where labels = 1
    for i, label in enumerate(labels):
        if label == 1:
            ax1.plot([i, i], [max_frequency_index - 5, max_frequency_index], color='r', linewidth=2)

    plt.tight_layout()
    plt.show()