from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
class Trainer:
    def __init__(self,train_loader, test_loader, criterion, optimizer, device):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device


    def trainGAT(self, epochs,model):
        model.to(self.device)
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for ndvi_seq, spatial_features, target in self.train_loader:
                ndvi_seq = ndvi_seq.to(self.device)
                spatial_features = spatial_features.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                predictions,_,_ = model(ndvi_seq, spatial_features)

                loss = self.criterion(predictions, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                
            print(f"Epoch {epoch + 1}/{epochs}, MSE: {total_loss / len(self.train_loader)}")

    def train(self, epochs,model):
        model.to(self.device)
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for ndvi_seq, spatial_features, target in self.train_loader:
                ndvi_seq = ndvi_seq.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                predictions = model(ndvi_seq)
                loss = self.criterion(predictions, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{epochs}, MSE: {total_loss / len(self.train_loader)}")

    def test(self, model):
        model.eval()
        preds = []
        tarLabels = []
        with torch.no_grad():
            total_loss = 0
            for ndvi_seq,_,target in self.test_loader:
                target = target.to(self.device)
                ndvi_seq = ndvi_seq.to(self.device)
                predictions = model(ndvi_seq)
                loss = self.criterion(predictions, target)
                total_loss += loss.item()
                preds.append(predictions.cpu().numpy())
                tarLabels.append(target.cpu().numpy())
        print(f"MSE: {total_loss / len(self.test_loader)}")
        rmse = np.sqrt(np.mean((np.array(tarLabels) - np.array(preds))** 2)) #convert to numpy
        print(f"RMSE: {rmse}")
        #Calculate mean absolute error
        mae = np.mean(np.abs(np.array(tarLabels) - np.array(preds)))
        print(f"MAE: {mae}")
        df = pd.DataFrame({'actual': np.array(tarLabels).flatten(), 'predicted': np.array(preds).flatten()})
        df.to_csv("gatgruindo16.csv")

    def testGAT(self, model):
        model.eval()
        preds = []
        tarLabels = []
        with torch.no_grad():
            total_loss = 0
            for ndvi_seq,spatial_features,target in self.test_loader:
                target = target.to(self.device)
                ndvi_seq = ndvi_seq.to(self.device)
                spatial_features = spatial_features.to(self.device)
                predictions,_,attention_weights = model(ndvi_seq, spatial_features)
                loss = self.criterion(predictions, target)
                total_loss += loss.item()
                preds.append(predictions.cpu().numpy())
                tarLabels.append(target.cpu().numpy())
        print(f"MSE: {total_loss / len(self.test_loader)}")
        rmse = np.sqrt(np.mean((np.array(tarLabels) - np.array(preds))** 2)) #convert to numpy
        print(f"RMSE: {rmse}")
        #Calculate mean absolute error
        mae = np.mean(np.abs(np.array(tarLabels) - np.array(preds)))
        print(f"MAE: {mae}")
        #self.plot_attention(attention_weights, 0, title="spatial Attention Weights")
        #save_average_attention(attention_weights)
        df = pd.DataFrame({'actual': np.array(tarLabels).flatten(), 'predicted': np.array(preds).flatten()})
        df.to_csv("gatindo16.csv")

    def plot_attention(self, attn_weights, head=0, title="Attention Heatmap"):
        attn = attn_weights[head].detach().cpu().numpy()
        plt.figure(figsize=(6, 5))
        sns.heatmap(attn, cmap="viridis")
        plt.title(f"{title} - Head {head}")
        plt.xlabel("Key Positions")
        plt.ylabel("Query Positions")
        plt.show()
    

    def train_with_attention_logging(self,epochs,model,log_interval=10):

        model.to(self.device)
        ndvi_fixed, spatial_fixed, _ = next(iter(self.test_loader))
        ndvi_fixed, spatial_fixed = ndvi_fixed.to(self.device), spatial_fixed.to(self.device)

        # Directory for saving attention plots
        log_dir = f"attention_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(log_dir, exist_ok=True)

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0

            for ndvi_batch, spatial_batch, target in self.train_loader:
                ndvi_batch = ndvi_batch.to(self.device)
                spatial_batch = spatial_batch.to(self.device)
                target = target.to(self.device)

                self.optimizer.zero_grad()
                pred, _, _ = model(ndvi_batch, spatial_batch)
                loss = self.criterion(pred.squeeze(), target)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
            print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

            # ---------- Log Attention ----------
            if epoch % log_interval == 0:
                model.eval()
                with torch.no_grad():
                    _, temporal_attn, spatial_attn = model(ndvi_fixed, spatial_fixed)

                # Save temporal heatmap for head 0
                plot_path_temp = os.path.join(log_dir, f"epoch_{epoch}_temporal.png")
                save_attention_plot(temporal_attn, head=0, title=f"Temporal Attention (Epoch {epoch})", save_path=plot_path_temp)

                # Save spatial heatmap for head 0
                plot_path_spatial = os.path.join(log_dir, f"epoch_{epoch}_spatial.png")
                save_attention_plot(spatial_attn, head=0, title=f"Spatial Attention (Epoch {epoch})", save_path=plot_path_spatial)
                plot_path_average = os.path.join(log_dir, f"epoch_{epoch}_spatial_average.png")
                save_average_attention(spatial_attn, head=0, title=f"Average Spatial Attention (Epoch {epoch})", save_path=plot_path_average)

        print(f"Attention plots saved in: {log_dir}")


# ---------- Save Attention Plot Function ----------

def save_attention_plot(attn_weights, head=0, title="Attention Heatmap", save_path=None):
    attn = attn_weights[0,head:,:].detach().cpu().numpy()
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(attn, cmap="viridis")
    ax.invert_yaxis() 
    plt.title(title)
    plt.xlabel("Key Positions")
    plt.ylabel("Query Positions")
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
def save_average_attention(attn_weights, head=0, title="Attention Heatmap", save_path=None):
    attn_matrix = attn_weights.reshape(16, 16)
    grid_attention = attn_matrix.mean(axis=0).detach().cpu().numpy()

    # Reshape back to 8x8 for visualization
    grid_attention_map = grid_attention.reshape(4, 4)
    plt.figure(figsize=(15, 12))
    sns.heatmap(grid_attention_map, cmap="magma", annot=True, fmt=".3f")
    plt.title("Average Attention Received per Grid Cell")

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_attention_weights(attn_weights, head=0, layer=0, title=None):
    """
    attn_weights: torch.Tensor of shape (num_layers, batch_size, num_heads, seq_len, seq_len)
    head: which head to visualize
    layer: which layer to visualize
    """
    # Extract the attention matrix for the specified layer & head
    attn = attn_weights[layer, 0, head].detach().cpu().numpy()  # (seq_len, seq_len)

    plt.figure(figsize=(5, 5))
    plt.imshow(attn, cmap="viridis")
    plt.colorbar()
    plt.title(title or f"Layer {layer+1} - Head {head+1}")
    plt.xlabel("Key Positions")
    plt.ylabel("Query Positions")
    plt.xticks(range(attn.shape[0]))
    plt.yticks(range(attn.shape[0]))
    plt.show()