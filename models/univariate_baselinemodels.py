# Baseline model implementations for NDVI forecasting on custom dataset
# Models: GRU, LSTM, 1D CNN, PatchTST-style Transformer

import torch
import torch.nn as nn
import math

# ----------- GRU Baseline ----------- #
class GRUBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1):
        super(GRUBaseline, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):  # x: [B, T, D]
        x = x.unsqueeze(-1)  # [B, T, D, 1]
        _, h = self.gru(x)  # h: [num_layers, B, hidden_dim]
        out = self.fc(h[-1])  # [B, output_dim]
        return out.view(-1,1)  # ensure shape [B, output_dim]


# ----------- LSTM Baseline ----------- #
class LSTMBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1):
        super(LSTMBaseline, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):  # x: [B, T, D]
        x = x.unsqueeze(-1)  # [B, T, D, 1]
        _, (h, _) = self.lstm(x)  # h: [num_layers, B, hidden_dim]
        out = self.fc(h[-1])  # [B, output_dim]
        return out.view(-1,1)  # ensure shape [B, output_dim]


# ----------- 1D CNN Baseline ----------- #
class CNN1DBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, output_dim=1):
        super(CNN1DBaseline, self).__init__()
        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):  # x: [B, T, D]
        x = x.unsqueeze(1)  # [B, T, D, 1]
        x = self.conv(x)        # [B, H, T]
        x = self.pool(x).squeeze(-1)  # [B, H]
        out = self.fc(x)  # [B, output_dim]
        return out.view(-1, 1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_layers, dropout=0.1, output_dim=1):
        super(TimeSeriesTransformer, self).__init__()
        # input_dim should be the number of features per time step (e.g., 1 for NDVI)
        self.input_proj = nn.Linear(input_dim, d_model)  # Map input feature dim to model dim
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):  # x: [B, T]
        # Project input feature dimension to d_model: [B, T, d_model]
        x = x.unsqueeze(-1)
        x = self.input_proj(x)
        # Add positional encoding: [B, T, d_model]
        x = self.positional_encoding(x)
        # Transformer expects input as [T, B, d_model]
        x = x.permute(1, 0, 2)
        # Pass through transformer encoder: [T, B, d_model]
        x = self.transformer(x)
        x = x[-1]
        # Take the output of the last time step for prediction: [B, d_model]
        out = self.fc(x)
        return out # [B, output_dim]

# ----------- PatchTST-inspired Variant ----------- #
class PatchTST(nn.Module):
    def __init__(self, input_dim, patch_size, d_model, n_heads, num_layers, dropout=0.1, output_dim=1):
        super(PatchTST, self).__init__()
        self.patch_size = patch_size
        self.linear_patch = nn.Linear(1, d_model) # Changed to accommodate 2D input
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):  # x: [B, T, D]
        B, T = x.shape  # Get batch size and sequence length
        num_patches = T // self.patch_size
        x = x[:, :num_patches * self.patch_size].reshape(B, num_patches, self.patch_size) # Reshape for 2D input
        x = self.linear_patch(x)  # [B, P, d_model]
        x = self.positional_encoding(x)
        x = x.permute(1, 0, 2)  # [P, B, d_model]
        x = self.transformer(x)
        out = self.fc(x[-1])
        return out  # [B, output_dim]


# ----------- Informer-like Variant (Simplified) ----------- #
class SimpleInformer(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_layers, dropout=0.1, output_dim=1):
        super(SimpleInformer, self).__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):  # x: [B, T, D]
        x = self.input_proj(x.unsqueeze(-1))  # [B, T, d_model]
        x = self.positional_encoding(x)
        x = x.permute(1, 0, 2)  # [T, B, d_model]
        encoded = self.encoder(x)  # [T, B, d_model]
        out = self.fc(encoded[-1])  # prediction from last token
        return out  # [B, output_dim]
