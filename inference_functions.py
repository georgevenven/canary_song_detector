import numpy as np
import matplotlib.pyplot as plt
import os 

def plot_spectrogram_and_labels(spec, labels, save_dir, file_name='spectrogram.png'):
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

    # Ensure the directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the figure
    plt.savefig(os.path.join(save_dir, file_name))

def replace_short_sequences(labels, target_label, min_length=50):
    """
    Replace short sequences of target labels with the opposite label.

    Args:
    - labels (numpy array): The 1D array of labels to process.
    - target_label (int): The label to target (either 0 or 1).
    - min_length (int): The minimum length of sequences to keep.

    Returns:
    - None. Modifies the input array in-place.
    """
    # Validate input
    if target_label not in [0, 1]:
        raise ValueError("target_label must be 0 or 1.")
        
    # Determine the opposite label
    opposite_label = 1 - target_label
    
    # Add padding to detect changes at the boundaries
    padded_labels = np.pad(labels, (1, 1), 'constant', constant_values=opposite_label)
    
    # Calculate the differences to find where the sequences of target_label's start and stop
    diff = np.diff(padded_labels)
    
    # Identify the start and stop indices for the target_label sequences
    start_indices = np.where(diff == (target_label - opposite_label))[0]
    stop_indices = np.where(diff == (opposite_label - target_label))[0]

    # Identify sequences that are shorter than `min_length`
    short_sequences = np.where((stop_indices - start_indices) < min_length)[0]
    
    # Replace short sequences with opposite_label
    for idx in short_sequences:
        start, stop = start_indices[idx], stop_indices[idx]
        labels[start:stop] = opposite_label

    return labels 

def pad_and_relabel(labels, target_label, n):
    """
    Pads occurrences of a target label by re-labeling an n number of spaces on each side of it.

    Args:
    - labels (numpy array): The 1D numpy array of labels to pad.
    - target_label (int): The label to target for padding.
    - n (int): Number of labels to pad on each side.

    Returns:
    - numpy array: A new array with padded labels.
    """
    # Create a copy to store the new labels, so we don't modify the original array
    new_labels = labels.copy()
    
    # Length of the labels array
    length = len(labels)
    
    for i, label in enumerate(labels):
        if label == target_label:
            # Determine the start and end indices for padding
            start = max(0, i - n)
            end = min(length, i + n + 1)  # +1 because Python slices are right-exclusive

            # Update the labels
            new_labels[start:end] = target_label
            
    return new_labels