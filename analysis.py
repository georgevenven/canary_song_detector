from sklearn.cluster import KMeans
import numpy as np
import torch
import umap.umap_ as umap
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle
from sklearn.metrics import f1_score
import pickle
import os 

class TweetyBERTAnalysis:
    def __init__(self, train_loader, model, device, num_classes=256):
        self.train_loader = train_loader
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.data_storage = {
            "ground_truth_labels": [],
            "predictions": [],
            "psuedo_labels": [],
            "spec": [],
            "cluster_labels": []
        }

    @staticmethod
    def one_hot_encode(labels, num_classes):
        one_hot = np.zeros((labels.shape[0], num_classes))
        one_hot[np.arange(labels.shape[0]), labels] = 1
        return one_hot

    def fit_kmeans(self, batch_limit=5):
        activations = []
        for i, (data, psuedo_labels, ground_truth_label) in enumerate(self.train_loader):
            if i > batch_limit:
                break
            predictions, _ = self.model.inference_forward(data.to(self.device), reduced_embedding=False)
            temp = predictions.view(-1, predictions.shape[-1])
            activations.append(temp.detach().cpu().numpy())

        k_means_arr = np.concatenate(activations, axis=0)
        self.kmeans = KMeans(n_clusters=2, random_state=0).fit(k_means_arr)

        return self.kmeans

    def collect_data(self, batch_limit=5):
        for i, (data, psuedo_labels, ground_truth_label) in enumerate(self.train_loader):
            if i > batch_limit:
                break
            predictions, _ = self.model.inference_forward(data.to(self.device), reduced_embedding=False)
            temp = predictions.view(-1, predictions.shape[-1])
            k_means_labels = self.kmeans.predict(temp.detach().cpu().numpy())
            k_means_labels = self.one_hot_encode(k_means_labels, self.num_classes)
            k_means_labels = k_means_labels.reshape(-1, 100, self.num_classes)
            k_means_labels = torch.Tensor(k_means_labels)
            k_means_labels = torch.argmax(k_means_labels, dim=-1).squeeze(1)

            spec = data.squeeze(1).permute(0,2,1)
            predictions = predictions.reshape(predictions.shape[0] * 10, 100, predictions.shape[2])
            spec = spec.reshape(spec.shape[0] * 10, 100, spec.shape[2])
            psuedo_labels = psuedo_labels.reshape(psuedo_labels.shape[0] * 10, 100, psuedo_labels.shape[2])
            ground_truth_label = ground_truth_label.reshape(ground_truth_label.shape[0] * 10, 100, ground_truth_label.shape[2])
            ground_truth_label = torch.argmax(ground_truth_label, dim=-1).squeeze(1)
            predictions = predictions.flatten(1,2)

            self.data_storage["spec"].append(spec.detach().cpu().numpy())
            self.data_storage["predictions"].append(predictions.detach().cpu().numpy())
            self.data_storage["ground_truth_labels"].append(ground_truth_label.cpu().numpy())
            self.data_storage["cluster_labels"].append(k_means_labels.cpu().numpy())

    def calculate_f1_score(self, batch_limit=20, k_means_batch_limit=2):
        predictions_arr = []
        ground_truth_labels_arr = []
        cluster_labels_arr = []
        
        for i in range(min(batch_limit, len(self.data_storage["predictions"]))):
            predictions = self.data_storage["predictions"][i]
            ground_truth_labels = self.data_storage["ground_truth_labels"][i]
            cluster_labels = self.data_storage["cluster_labels"][i]
            
            # Reshape to flatten the labels
            ground_truth_labels = ground_truth_labels.reshape(-1)
            cluster_labels = cluster_labels.reshape(-1)
            
            ground_truth_labels_arr.append(ground_truth_labels)
            cluster_labels_arr.append(cluster_labels)
        
        ground_truth_labels = np.concatenate(ground_truth_labels_arr, axis=0)
        cluster_labels = np.concatenate(cluster_labels_arr, axis=0)
        
        # Modify cluster labels based on the logic
        cluster_labels[(cluster_labels == 0)] = 0
        cluster_labels[(cluster_labels == 1)] = 1
        
        # Compute F1 Score
        f1 = f1_score(ground_truth_labels, cluster_labels, average='weighted')
        return f1

    def plot_umap(self, output_dir, plot_ground_truth=False):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        predictions = np.concatenate(self.data_storage['predictions'], axis=0)
        ground_truth_labels = np.concatenate(self.data_storage['ground_truth_labels'], axis=0)
        cluster_labels = np.concatenate(self.data_storage['cluster_labels'], axis=0)

        num_unique_clusters = len(np.unique(cluster_labels))
        new_cmap = cm.get_cmap('jet', num_unique_clusters)
        
        # Using the same color map for ground truth
        label_to_color = {label: new_cmap(i / num_unique_clusters) for i, label in enumerate(np.unique(ground_truth_labels))}
        
        cluster_to_color = {label: new_cmap(i / num_unique_clusters) for i, label in enumerate(np.unique(cluster_labels))}

        colors_for_clusters = [np.mean([cluster_to_color[int(lbl)] for lbl in row], axis=0) for row in cluster_labels]

        reducer = umap.UMAP(random_state=42, n_neighbors=20, min_dist=0.1, n_components=2, metric='euclidean')
        embedding_outputs = reducer.fit_transform(predictions)

        if plot_ground_truth:
            colors_for_points = [np.mean([label_to_color[int(lbl)] for lbl in row], axis=0) for row in ground_truth_labels]
            plt.figure(figsize=(8, 6))
            plt.scatter(embedding_outputs[:, 0], embedding_outputs[:, 1], s=5, c=colors_for_points)
            plt.title('UMAP projection of the TweetyBERT (Ground Truth)', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'umap_ground_truth.png'))
            plt.close()

        plt.figure(figsize=(8, 6))
        plt.scatter(embedding_outputs[:, 0], embedding_outputs[:, 1], s=5, c=colors_for_clusters)
        plt.title('UMAP projection of the TweetyBERT (K-means Labels)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'umap_kmeans_labels.png'))
        plt.close()