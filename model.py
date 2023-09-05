import torch
import torch.nn as nn
import torch.nn.functional as F

class ModifiedTransformerEncoder(nn.TransformerEncoder):
    def forward(self, src, mask=None, src_key_padding_mask=None):
        """ Modified forward function to return outputs of all layers """
        output = src
        outputs = []

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            outputs.append(output)

        return outputs

class TweetyBERT(nn.Module):
    def __init__(self, d_transformer, nhead_transformer, embedding_dim, num_labels, tau=0.1, dropout=0.1, transformer_layers=3, dim_feedforward=128, reduced_embedding = 4):
        super(TweetyBERT, self).__init__()
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
        self.transformerProjection = nn.Linear(448, d_transformer)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_transformer, nhead=nhead_transformer, batch_first=True, dim_feedforward=dim_feedforward)
        self.transformer_encoder = ModifiedTransformerEncoder(self.encoder_layer, num_layers=transformer_layers)
        
        self.transformerDeProjection = nn.Linear(d_transformer, embedding_dim)

        # label embedding
        self.label_embedding = nn.Embedding(self.num_labels, embedding_dim)

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
        # Collect outputs of all layers
        all_layer_outputs = self.transformer_encoder(x)
        final_output = all_layer_outputs[-1]  # The last output is the final one
        
        return final_output, all_layer_outputs


    def masking_operation(self, x, p=0.00, mask_value=0.0, m=10):
        batch, dim, length = x.size()

        # Randomly determine which elements to mask based on probability p
        prob_mask = torch.rand(length, device=x.device) < p

        # Expand the mask for m subsequent elements
        expanded_mask = torch.zeros_like(prob_mask)
        for i in range(length):
            if prob_mask[i]:
                expanded_mask[i:min(i+m, length)] = 1

        mask = expanded_mask.view(1, 1, -1).expand_as(x)

        # Invert the mask and convert it to float
        mask_float = (~mask).float()

        # Apply the mask
        x = x * mask_float

        return x, mask_float

    def train_forward(self, x):
        x = self.feature_extractor_forward(x)
        x, mask = self.masking_operation(x)
        x, all_layers = self.transformer_forward(x)
        x = self.transformerDeProjection(x)
        return x, mask

    def inference_forward(self, x, reduced_embedding=False):
        x = self.feature_extractor_forward(x)
        x, layers = self.transformer_forward(x)

        if reduced_embedding == False:
            x = self.transformerDeProjection(x)
        else:
            x = self.reduced_embedding(x)

        return x, layers

    # Add the following method to the HuBERT class

    def compute_probabilities(self, predictions, targets):
        """
        Compute the softmax probabilities based on cosine similarity.

        Input Params:
        - Targets (tensor): Shape of targets (batch, seq, embedding) -> last dim is 1
        - Predictions (tensor): Shape of predictions of the model (batch, seq, embedding) -> last dim is embedding size

        Output Params:
        - Probabilities (tensor): Shape of softmax of values of correct target and prediction ()
        """

        # Convert targets to shape [batch x seq]
        targets = targets.argmax(dim=-1)
        # Project to embedding size
        targets = self.label_embedding(targets)

        # Prepare all_labels tensor of shape [num_labels, embedding]
        all_labels = torch.arange(self.num_labels).to(predictions.device)
        all_labels = self.label_embedding(all_labels)

        # Compute cosine similarities for all sequences and all labels
        # Shape will be [batch, seq, num_labels]
        csim = torch.einsum('bse,le->bsl', predictions, all_labels)
        csim /= (predictions.norm(dim=-1, keepdim=True) * all_labels.norm(dim=-1, keepdim=True).T)

        # Exponentiate the cosine similarities
        exp_csim = torch.exp(csim)

        # Exponentiate the cosine similarities for the correct targets
        # Using gather to pick the right label for each sequence in the batch
        correct_indices = targets.argmax(dim=-1).unsqueeze(-1)
        exp_csim_correct = torch.gather(exp_csim, -1, correct_indices).squeeze(-1)

        # Sum across the last dimension to get the normalization factor
        sum_exp_csim = exp_csim.sum(-1)

        # Compute the final probabilities
        prob_seq = exp_csim_correct / sum_exp_csim

        return prob_seq


    def compute_loss(self, predictions, targets, mask, alpha=0):
        """
        Compute the cross-entropy loss for the masked and unmasked positions separately,
        and then combine them using a geometric mean.
        Mask, shape [batch, features, seq], 0 means masked, 1 means unmasked
        alpha of 1 means all loss is concentrated in the masked portion 
        """

        # Determine masked and unmasked indices
        masked_indices = mask[:, 0, :] == 0.0

        unmasked_indices = ~masked_indices

        # Compute softmax probabilities
        probabilities = self.compute_probabilities(predictions=predictions, targets=targets)

        # Compute log probabilities
        log_probabilities = torch.log(probabilities + 1e-10)

        # Compute cross-entropy loss
        loss = -log_probabilities

        # # Compute loss for masked and unmasked positions separately
        if masked_indices.sum() > 0:
            masked_loss = loss[masked_indices].mean()
        else:
            masked_loss = 0.0  # or some other default value

        if unmasked_indices.sum() > 0:
            unmasked_loss = loss[unmasked_indices].mean()
        else:
            unmasked_loss = 0.0  # or some other default value

        # Combine losses with the geometric mean
        epsilon = 1e-10  # A small constant to prevent zero loss
        masked_loss += epsilon
        unmasked_loss += epsilon
        combined_loss = alpha * masked_loss + (1 - alpha) * unmasked_loss

        return combined_loss


    def cross_entropy_loss(self, y_pred, y_true):
        """loss function for TweetyNet
        Parameters
        ----------
        y_pred : torch.Tensor
            output of TweetyNet model, shape (batch, classes, timebins)
        y_true : torch.Tensor
            one-hot encoded labels, shape (batch, classes, timebins)
        Returns
        -------
        loss : torch.Tensor
            mean cross entropy loss
        """
        loss = nn.CrossEntropyLoss()
        return loss(y_pred, y_true)

    def binary_cross_entropy(self, y_pred, y_true):
        criterion = nn.BCELoss()
        loss  = criterion(y_pred, y_true)
        return loss 