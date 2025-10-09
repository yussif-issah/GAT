import torch
import torch.nn as nn

class GridAttentionTransformer(nn.Module):
    def __init__(self, seq_len, d_model, n_heads, n_layers, dropout=0.1,n_cells=64):
        super(GridAttentionTransformer, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # Temporal encoder: encodes NDVI sequences (per cell)
        self.temporal_weights =  nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, dropout=dropout,batch_first=True)

        self.temporal_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, dropout=dropout),
            num_layers=1
        )
        self.temporal_fc = nn.Linear(1, d_model)  # Input NDVI is scalar

        #positional encoding
        self.cell_pos_embed = nn.Parameter(torch.randn(n_cells, d_model))
        self.time_pos_embed = nn.Parameter(torch.randn(seq_len, d_model))

        # Spatial embedding (positional or auxiliary features)
        self.spatial_fc = nn.Linear(10, d_model)

        # Cross-cell attention

        self.spatial_weights = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, dropout=dropout,batch_first=True)

        self.cross_cell_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, dropout=dropout),
            num_layers=n_layers
        )

        self.lstm_head = nn.LSTM(
          input_size=d_model,
          hidden_size=d_model // 2,  # You can choose the hidden size
          num_layers=1,
          batch_first=True,
          bidirectional=False  # You can experiment with bidirectional too
        )
        self.fc_out = nn.Linear(d_model // 2, 1)  # Match LSTM hidden output

    def forward(self, ndvi_seqs, spatial_positions):
        """
        ndvi_seqs: (B, T)  -- NDVI sequences for B grid cells
        spatial_positions: (B, 10) -- spatial metadata per cell
        """
        B, T = ndvi_seqs.shape

        # Reshape and embed NDVI: (B, T, 1) → (T, B, d_model)
        x = self.temporal_fc(ndvi_seqs.unsqueeze(-1))         # (B, T, d_model)
        # Add 2D positional encoding
        pos = self.cell_pos_embed.unsqueeze(1) + self.time_pos_embed.unsqueeze(0)  # (B, T, d_model)
        x = x + pos
        x = x.permute(1, 0, 2)                                # (T, B, d_model)

        # Temporal encoding (within each cell)
        _,temporal_weights = self.temporal_weights.self_attn(x,x,x,need_weights=True)  # (T, B, d_model)

        temp_encoded = self.temporal_encoder(x)               # (T, B, d_model)
        cell_embeddings = temp_encoded[-1]                     # (B, d_model) ← last time step for each cell

        # Spatial embeddings
        spatial_embed = self.spatial_fc(spatial_positions)    # (B, d_model)

        # Combine spatial + temporal
        combined_embed = cell_embeddings + spatial_embed      # (B, d_model)

        # Now treat cells as a sequence → shape (B, d_model) → (B, 1, d_model)
        spatial_input = combined_embed.unsqueeze(1)           # (B, 1, d_model)
        spatial_input = spatial_input.permute(1, 0, 2)        # (1, B, d_model)

        # Cross-cell attention (across grid cells)
        _,spatial_weights = self.spatial_weights.self_attn(spatial_input, spatial_input, spatial_input, need_weights=True)  # (1, B, d_model)
        spatial_output = self.cross_cell_encoder(spatial_input)  # (1, B, d_model)
        lstm_input = spatial_output.permute(1, 0, 2)             # (B, d_model)
        lstm_out, _ = self.lstm_head(lstm_input)

        # Get the output at last time step
        final_embedding = lstm_out[:, -1, :]  # (B, hidden_size)

        # Final prediction
        prediction = self.fc_out(final_embedding)  # (B, 1)               # (B, 1)
        return prediction, temporal_weights, spatial_weights

class TransformerEncoderLayerWithAttn(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None, need_weights=True):
        attn_output, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=need_weights
        )
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff)
        src = self.norm2(src)
        return src, attn_weights  # <--- returning both


# ---------- Main Model with Attention Capture ----------
class GridAttentionTransformers(nn.Module):
    def __init__(self, seq_len, d_model, n_heads, n_layers, dropout=0.1, n_cells=64):
        super(GridAttentionTransformers, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # Temporal encoder
        self.temporal_fc = nn.Linear(1, d_model)
        self.temporal_encoder = TransformerEncoderLayerWithAttn(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, dropout=dropout
        )

        # Positional embeddings
        self.cell_pos_embed = nn.Parameter(torch.randn(n_cells, d_model))
        self.time_pos_embed = nn.Parameter(torch.randn(seq_len, d_model))

        # Spatial embedding
        self.spatial_fc = nn.Linear(10, d_model)

        # Cross-cell attention
        self.cross_cell_encoder = TransformerEncoderLayerWithAttn(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, dropout=dropout
        )

        # LSTM head
        self.lstm_head = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.fc_out = nn.Linear(d_model // 2, 1)

    def forward(self, ndvi_seqs, spatial_positions):
        B, T = ndvi_seqs.shape

        # NDVI embedding + positional encoding
        x = self.temporal_fc(ndvi_seqs.unsqueeze(-1))  # (B, T, d_model)
        pos = self.cell_pos_embed.unsqueeze(1) + self.time_pos_embed.unsqueeze(0)
        x = x + pos
        x = x.permute(1, 0, 2)  # (T, B, d_model)

        # Temporal encoding
        temp_encoded, temporal_attn_weights = self.temporal_encoder(x)
        cell_embeddings = temp_encoded[-1]  # (B, d_model)

        # Spatial embeddings
        spatial_embed = self.spatial_fc(spatial_positions)  # (B, d_model)

        # Combine
        combined_embed = cell_embeddings + spatial_embed

        # Prepare for cross-cell attention
        spatial_input = combined_embed.unsqueeze(1)  # (1, B, d_model)

        # Cross-cell attention
        spatial_output, spatial_attn_weights = self.cross_cell_encoder(spatial_input)
        lstm_input = spatial_output.permute(1, 0, 2)
        lstm_out, _ = self.lstm_head(lstm_input)

        final_embedding = lstm_out[:, -1, :]
        prediction = self.fc_out(final_embedding)

        return prediction, temporal_attn_weights, spatial_attn_weights