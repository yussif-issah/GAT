from models.univariate_baselinemodels import CNN1DBaseline, GRUBaseline, LSTMBaseline, PatchTST, SimpleInformer, TimeSeriesTransformer
from training.training import Trainer
from data_preprocessing.preprocess import DataPreprocessor
from data_preprocessing.dataloaders import CustomDataLoader
from models.gat import GridAttentionTransformer
import time

import numpy as np
import random
import torch

seed = 5
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



DATA_PATH = 'data/indo_indices240_2000_2023.csv'
GRID_SIZE = 240
WINDOW_SIZE = 5
STEP_SIZE = 1
FORECAST_HORIZON = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#process data
indices = ['MNDWI','NDMI', 'NDWI', 'NBR', 'EVI', 'SAVI']  # Example additional indices
dataPreProcessor = DataPreprocessor(file_path=DATA_PATH, grid_size=GRID_SIZE, window_size=WINDOW_SIZE, step_size=STEP_SIZE, forecast_horizon=FORECAST_HORIZON)
sequences_all, spatial_features_all, targets_all = dataPreProcessor.create_sequences(multi_index=False, indices=indices)
data_loader = CustomDataLoader(sequences_all, spatial_features_all, targets_all, batch_size=GRID_SIZE, train_size=3840, test_size=720) #16 rain_size=256, test_size=48 #64 train_size=1024, test_size=192 #240 train_size=3840, test_size=720
train_loader, test_loader = data_loader.getDataLoaders()


#initialize model, criterion and optimizer
gat = GridAttentionTransformer(n_cells=GRID_SIZE,seq_len=5,d_model=16,n_heads=4,n_layers=2,dropout=0.005)
gru = GRUBaseline(input_dim=1,hidden_dim=32, num_layers=8,output_dim=1)
lstm = LSTMBaseline(input_dim=1,hidden_dim=32, num_layers=8,output_dim=1)
cnn = CNN1DBaseline(input_dim=1,hidden_dim=32, kernel_size=5,output_dim=1)
pst = PatchTST(input_dim=1, patch_size=1,d_model=16, n_heads=4, num_layers=2, output_dim=1)
sit= SimpleInformer(input_dim=1, d_model=16, n_heads=4, num_layers=4, output_dim=1)
tst = TimeSeriesTransformer(input_dim=1,d_model=16, n_heads=4, num_layers=4, output_dim=1)

names = ['gat', 'gru', 'lstm', 'cnn', 'pst', 'sit', 'tst']
models= [gat, gru, lstm, cnn, pst, sit, tst]


criterion = torch.nn.MSELoss()


for name, model in zip(names, models):
    #train model
    optimizer = torch.optim.Adam(lr=1e-3, betas=(0.9, 0.98), eps=1e-9, params=model.parameters())
    trainer = Trainer(train_loader, test_loader, criterion, optimizer, device)
    print(f"Training {name} model...")
    if name == "gat":
        start = time.time()
        trainer.trainGAT(50,model=model)
        end = time.time()
        print(f"Training {name} model took {end - start:.2f} seconds.")
        print(f" Testing {name} model...")
        trainer.testGAT(model)

    else:
        trainer.train(epochs=50, model=model)
        print(f"Testing {name} model...")
        trainer.test(model)





