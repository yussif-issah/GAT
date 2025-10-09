from data_preprocessing.dataset import SlidingWindowNDVIDataset
import torch
from torch.utils.data import DataLoader

class CustomDataLoader:
    def __init__(self,sequences_all, spatial_features_all, targets_all, batch_size=64,train_size=100,test_size=20):
        self.dataset = SlidingWindowNDVIDataset(sequences_all, spatial_features_all, targets_all)
        self.batch_size = batch_size
        self.train_size = train_size
        self.test_size = test_size
    
    def getDataLoaders(self):
        #train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_dataset = torch.utils.data.Subset(self.dataset, range(self.train_size)) # Use Subset for sequential split
        test_dataset = torch.utils.data.Subset(self.dataset, range(self.train_size, self.train_size + self.test_size))  # Use Subset for sequential split
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader =  DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader