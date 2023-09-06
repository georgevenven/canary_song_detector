import os
import scipy.io
import numpy as np
import torch 
import matplotlib.pyplot as plt
from matplotlib import cm

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

def plot_spectrogram_and_labels(spec, labels):
    labels = np.argmax(labels, axis=-1)

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