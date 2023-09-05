
import os
import shutil
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans, KMeans
from tqdm import tqdm
import matplotlib.pyplot as plt
from umap_pytorch import PUMAP
from umap_pytorch import load_pumap
import umap.umap_ as umap

class SpectrogramProcessor:
    def clear_directory(self, directory_path):
        """Deletes all files and subdirectories within a specified directory."""
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    
    def __init__(self, data_root, train_dir, test_dir, n_clusters, embedding_dir='embedding.pkl', reducer_dir='reducer.pkl', train_prop=0.8, limit=10, segment_size=1000, overlap=.25):
        self.data_root = data_root
        self.n_clusters = n_clusters
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.train_prop = train_prop
        self.limit = limit
        self.segment_size = segment_size
        self.overlap = overlap
        self.embedding_dir = embedding_dir
        self.reducer_dir = reducer_dir 
        self.n_clusters = n_clusters
        self.kmeans = MiniBatchKMeans(n_clusters=self.n_clusters)
 
        # Create directories if they don't exist
        if not os.path.exists(self.train_dir):
            os.mkdir(self.train_dir)

        if not os.path.exists(self.test_dir):
            os.mkdir(self.test_dir)

    def generate_train_test(self):
        files = os.listdir(self.data_root)
        for i, file in enumerate(tqdm(files, desc="Processing files")):
            if file.endswith(".npz"):
                f = np.load(os.path.join(self.data_root, file))
                spectogram = f['s']
                labels = f['labels']
                new_labels = f['labels']

                # Create a new dictionary with only the required attributes
                f_dict = {'s': spectogram, 'labels': labels, 'new_labels': new_labels}

                step_size = int(self.overlap * self.segment_size)
                length = int(spectogram.shape[1] - (spectogram.shape[1] % self.segment_size))

                # Break the spectrogram into overlapping segments
                for idx in range(0, length, step_size):
                    if idx + self.segment_size >= length:
                        continue
                    segment = spectogram[:, idx:idx + self.segment_size]
                    segment_labels = labels[:, idx:idx + self.segment_size]

                    segment_filename = f"{file[:-4]}_segment_{idx}.npz"
                    
                    # Update the segment and segment_labels in the f_dict
                    f_dict['s'] = segment[8:500,:]
                    f_dict['labels'] = segment_labels
                    f_dict['new_labels'] = np.zeros(segment_labels.shape)

                    # Decide where to save the segmented file
                    if np.random.uniform() < self.train_prop:
                        save_path = os.path.join(self.train_dir, segment_filename)
                    else:
                        save_path = os.path.join(self.test_dir, segment_filename)

                    np.savez(save_path, **f_dict)

    def generate_embedding(self, samples=3e5, time_bins_per_sample=100, reduction_dims=2):
        files = os.listdir(self.train_dir)
        all_spectograms = []
        for i, file in enumerate(files):
            if file.endswith(".npz"):
                f = np.load(os.path.join(self.train_dir, file))
                spectogram = f['s'].T
                length = spectogram.shape[0]
                num_times = length // time_bins_per_sample
                spectogram = spectogram.reshape(int(num_times) * 1, spectogram.shape[1]  * time_bins_per_sample)
                all_spectograms.append(spectogram)

            if len(all_spectograms) > samples:
                break

        arr = np.concatenate(all_spectograms, axis=0)
        arr = torch.Tensor(arr)
        means = arr.mean(dim=1, keepdim=True)
        stds = arr.std(dim=1, keepdim=True)
        arr = (arr - means) / (stds + 1e-7)
        pumap = PUMAP(epochs=30, lr=1e-3, match_nonparametric_umap=True, n_components=reduction_dims)
        pumap.fit(arr)
        embedding_pumap = pumap.transform(arr)

        with open(self.embedding_dir, 'wb') as f:
            torch.save(embedding_pumap, f)

        pumap.save(self.reducer_dir)

    def generate_logit_embedding(self, samples=300, time_bins_per_sample=1):
        files = os.listdir(self.train_dir)
        processed_samples = 0

        for i, file in enumerate(files):
            if file.endswith(".npz"):
                f = np.load(os.path.join(self.train_dir, file))

                spectogram = f['logits'].T

                length = spectogram.shape[0]
                num_times = length // time_bins_per_sample
                spectogram = spectogram.reshape(int(num_times), -1)

                # Compute means and standard deviations
                means = np.mean(spectogram, axis=1, keepdims=True)
                stds = np.std(spectogram, axis=1, keepdims=True)

                # Standardize the spectogram
                spectogram = (spectogram - means) / (stds + 1e-7)

                # Partial fit with MiniBatchKMeans
                self.kmeans.partial_fit(spectogram)
                print(f"File {i+1}/{len(files)} processed.")

                processed_samples += 1
                if processed_samples >= samples:
                    break

        print("Final model fit done.")

    def generate_labels(self, embedding):
        kmeans = KMeans(n_clusters=self.n_clusters)
        labels_kmeans = kmeans.fit_predict(embedding)
        return labels_kmeans, kmeans

    def plot_embedding_and_labels(self):
        # Load the embeddings
        with open(self.embedding_dir, 'rb') as f:
            embedding_pumap = torch.load(f)

        # Get labels from the generate_labels function
        labels_kmeans, _ = self.generate_labels(embedding_pumap)

        # Plotting
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(embedding_pumap[:, 0], embedding_pumap[:, 1], c=labels_kmeans, cmap='viridis', s=5)
        plt.colorbar(scatter)
        plt.title("PUMAP Embeddings with K-means Labels")
        plt.xlabel("Embedding Dimension 1")
        plt.ylabel("Embedding Dimension 2")
        plt.show()

    def generate_train_test_labels(self, time_bins_per_sample=100, logits=False, reduce=True):
        # Load the reducer
        reducer = load_pumap(self.reducer_dir)
        with open(self.embedding_dir, 'rb') as f:
            embedding_pumap = torch.load(f)

        labels, kmeans = self.generate_labels(embedding_pumap)
        # For each file in /train 
        for file in tqdm(os.listdir(self.train_dir), desc="Processing Train Files"):  # Wrapped os.listdir with tqdm
            if file.endswith(".npz"):
                f = np.load(os.path.join(self.train_dir, file))

                if logits == False:
                    spectogram = f['s'].T
                else:
                    spectogram = f['logits'].T

                original_spectogram = f['s']
                length = spectogram.shape[0]
                num_times = length // time_bins_per_sample
                spectogram = spectogram.reshape(int(num_times) * 1, spectogram.shape[1]  * time_bins_per_sample)
                means = np.mean(spectogram, axis=1, keepdims=True)
                stds = np.std(spectogram, axis=1, keepdims=True)
                spectogram = (spectogram - means) / (stds + 1e-7)
                spectogram = reducer.transform(torch.Tensor(spectogram))
                spectogram = np.array(spectogram)
                labels = kmeans.predict(spectogram)
                repeated_labels = np.repeat(labels, time_bins_per_sample)
                repeated_labels = repeated_labels.reshape(1,-1)
                f_dict = {'s': original_spectogram, 'labels': f['labels'], 'new_labels': repeated_labels}
                save_path = os.path.join(self.train_dir, file)
                np.savez(save_path, **f_dict)

        for file in tqdm(os.listdir(self.test_dir), desc="Processing Test Files"):  # Wrapped os.listdir with tqdm
            if file.endswith(".npz"):
                f = np.load(os.path.join(self.test_dir, file))

                if logits == False:
                    spectogram = f['s'].T
                else:
                    spectogram = f['logits'].T

                original_spectogram = f['s']
                length = spectogram.shape[0]
                num_times = length // time_bins_per_sample
                spectogram = spectogram.reshape(int(num_times) * 1, spectogram.shape[1] * time_bins_per_sample)
                means = np.mean(spectogram, axis=1, keepdims=True)
                stds = np.std(spectogram, axis=1, keepdims=True)
                spectogram = (spectogram - means) / (stds + 1e-7)
                spectogram = reducer.transform(torch.Tensor(spectogram))
                spectogram = np.array(spectogram)
                labels = kmeans.predict(spectogram)
                repeated_labels = np.repeat(labels, time_bins_per_sample)
                repeated_labels = repeated_labels.reshape(1,-1)
                f_dict = {'s': original_spectogram, 'labels': f['labels'], 'new_labels': repeated_labels}
                save_path = os.path.join(self.test_dir, file)
                np.savez(save_path, **f_dict)


    def generate_logits_train_test_labels(self, time_bins_per_sample=1, logits=True):
        # Handle training data
        train_files = [file for file in os.listdir(self.train_dir) if file.endswith(".npz")]
        total_train_files = len(train_files)
        print(f"Processing {total_train_files} training files...")
        
        for i, file in enumerate(train_files):
            self.process_file(file, self.train_dir, time_bins_per_sample, logits)
            print(f"Processed training file {i + 1}/{total_train_files}")

        # Handle test data
        test_files = [file for file in os.listdir(self.test_dir) if file.endswith(".npz")]
        total_test_files = len(test_files)
        print(f"Processing {total_test_files} test files...")
        
        for i, file in enumerate(test_files):
            self.process_file(file, self.test_dir, time_bins_per_sample, logits)
            print(f"Processed test file {i + 1}/{total_test_files}")

    def process_file(self, file, directory, time_bins_per_sample, logits):
        f = np.load(os.path.join(directory, file))

        if not logits:
            spectogram = f['s'].T
        else:
            spectogram = f['logits'].T

        original_spectogram = f['s']
        length = spectogram.shape[0]
        num_times = length // time_bins_per_sample
        spectogram = spectogram.reshape(int(num_times) * 1, spectogram.shape[1] * time_bins_per_sample)
        means = np.mean(spectogram, axis=1, keepdims=True)
        stds = np.std(spectogram, axis=1, keepdims=True)
        spectogram = (spectogram - means) / (stds + 1e-7)

        # Convert to double precision
        spectogram = np.array(spectogram)
        
        # Predict labels
        labels = self.kmeans.predict(spectogram)

        f_dict = {'s': original_spectogram, 'labels': f['labels'], 'new_labels': labels, 'logits': spectogram}
        save_path = os.path.join(directory, file)
        np.savez(save_path, **f_dict)