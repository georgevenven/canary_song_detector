import os
import torch
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

def orthogonal_regularization_term(x, scaling_factor=15):
        xT_x = torch.mm(x.T, x)  # should be close to identity
        return scaling_factor * ((xT_x - torch.eye(xT_x.size(0), device=x.device)).norm(p='fro'))

class ModelTrainer:
    def __init__(self, model, train_loader, test_loader, optimizer, device, max_steps=10000, eval_interval=500, save_interval=1000, save_dir='saved_models'):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = device
        self.max_steps = max_steps
        self.eval_interval = eval_interval
        self.scheduler = StepLR(optimizer, step_size=10000, gamma=1)

        self.loss_list = []
        self.val_loss_list = []
        self.sum_squared_weights_list = []

        self.save_interval = save_interval
        self.save_dir = save_dir

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def sum_squared_weights(self):
        sum_of_squares = sum(torch.sum(p ** 2) for p in self.model.parameters())
        return sum_of_squares   

    def validate_model(self):
        self.model.eval()
        total_val_loss = 0
        num_val_batches = 0
        with torch.no_grad():
            for i, (spec, label, _) in enumerate(self.test_loader):
                if i > 1:
                    break
                spec = spec.to(self.device)
                label = label.to(self.device)
                output, mask = self.model.train_forward(spec)
                similarity_loss = self.model.compute_loss(output, label, mask)

                loss = similarity_loss

                total_val_loss += loss.item()
                num_val_batches += 1

        return total_val_loss / num_val_batches

    def train(self):
        step = 0
        train_iter = iter(self.train_loader)
        lambda_reg = 0.00

        spec, label, _ = next(train_iter)
        spec = spec.to(self.device)
        label = label.to(self.device)
        output, mask = self.model.train_forward(spec)
        
        similarity_loss = self.model.compute_loss(output, label, mask)
        initial_loss = similarity_loss

        l1_norm = torch.norm(self.model.transformerDeProjection.weight, p=1)
        initial_loss += lambda_reg * l1_norm

        print(f'Initial Loss: {initial_loss.item():.2e}')
        self.loss_list.append(initial_loss.item())

        while step < self.max_steps:
            try:
                spec, label, _ = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                spec, label, _ = next(train_iter)

            spec = spec.to(self.device)
            label = label.to(self.device)

            output, mask = self.model.train_forward(spec)
            similarity_loss = self.model.compute_loss(output, label, mask)
            loss =  similarity_loss

            # Compute orthogonal regularization terms
            ortho_reg_transformer_proj = orthogonal_regularization_term(self.model.transformerProjection.weight)
            ortho_reg_transformer_deproj = orthogonal_regularization_term(self.model.transformerDeProjection.weight)

            # Add the orthogonal regularization terms to the loss
            loss += lambda_reg * (ortho_reg_transformer_proj + ortho_reg_transformer_deproj)  # Modified line

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if step % self.eval_interval == 0:
                avg_val_loss = self.validate_model()
                self.val_loss_list.append(avg_val_loss)
                self.loss_list.append(loss.item())
                self.sum_squared_weights_list.append(self.sum_squared_weights().item())
                print(f'Step [{step}/{self.max_steps}], Training Loss: {loss.item():.4e}, Validation Loss: {avg_val_loss:.4e}')

            if step % self.save_interval == 0:
                self.save_model(step)

            step += 1

    def save_model(self, step):
        filename = f"model_step_{step}.pth"
        filepath = os.path.join(self.save_dir, filename)
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def plot_results(self):
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(self.loss_list, label='Training Loss')
        plt.plot(self.val_loss_list, label='Validation Loss')
        plt.legend()
        plt.title('Training and Validation Loss')

        plt.subplot(1, 2, 2)
        plt.plot(self.sum_squared_weights_list, color='red', label='Sum of Squared Weights')
        plt.legend()
        plt.title('Sum of Squared Weights per Step')

        plt.tight_layout()
        plt.show()