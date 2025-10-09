from torch.utils.data import Dataset

class SlidingWindowNDVIDataset(Dataset):
    def __init__(self, ndvi_sequences, spatial_positions, targets):
      self.ndvi_sequences = ndvi_sequences
      self.spatial_positions = spatial_positions
      self.targets = targets

    def __len__(self):
      return len(self.targets)

    def __getitem__(self, idx):
      return self.ndvi_sequences[idx], self.spatial_positions[idx], self.targets[idx]