import json 
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, file_path,grid_size=16,window_size=5,step_size=1,forecast_horizon=1):
        self.file_path = file_path
        self.grid_size = grid_size
        self.window_size = window_size
        self.step_size = step_size
        self.forecast_horizon = forecast_horizon
        self.data = None
        self.ndvi_series = None
        self.data = self.preprocessUnivariateFile()


    def preprocessUnivariateFile(self):
        data = pd.read_csv(self.file_path)
        data['strGeom']= data['.geo'].apply(lambda x: x.replace("'", '"'))
        data['geomArray'] = data['strGeom'].apply(json.loads)
        data['coords'] = data['geomArray'].apply(lambda x: x['coordinates'])
        data['coordinatesArray'] = data['coords'].apply(lambda x: np.array([coord for sublist in x for coord in sublist]).flatten())
        data['year'] = data['Year'].astype(int)
        df = data.drop(columns=['strGeom', 'geomArray', 'coords','system:index','Year','.geo'],axis=1)

        df['grid_id'] = data.index % self.grid_size
        df.fillna(method='ffill', inplace=True)
        return df

    def create_sliding_windows(self,ndvi_series, spatial_position):
        sequences = []
        targets = []
        spatial_features = []
        # Generate sequences
        for start_idx in range(0, len(ndvi_series) - self.window_size - self.forecast_horizon+1, self.step_size):
            # Extract NDVI window and target
            ndvi_window = ndvi_series[start_idx : start_idx + self.window_size]
            target_value = ndvi_series[start_idx + self.window_size + self.forecast_horizon - 1]
            sequences.append(ndvi_window)
            targets.append(target_value)
            spatial_features.append(spatial_position)  # Same spatial position for each window

        return np.array(sequences), np.array(targets), np.array(spatial_features)
    
    def create_sequences(self,multi_index=False,indices=[]):
        sequences_all = []
        targets_all = []
        spatial_features_all = []

        if multi_index:
            for i in range(self.grid_size):
                grid_data = self.data[self.data['grid_id'] == i]
                ndvi_series = grid_data['NDVI'].values
                spatial_coordinates = grid_data['coordinatesArray'].iloc[0][:10]
                final_seq,ndvi_targets,spatial_features = self.create_sliding_windows(
                    ndvi_series, spatial_coordinates
                )
                for index in indices:
                    series = grid_data[index].values
                    #ndvi_series = np.concatenate((ndvi_series,series),axis=0)
                    sequences, targets, spatial_features = self.create_sliding_windows(
                        series, spatial_coordinates
                    )
                    final_seq = np.concatenate((final_seq, sequences), axis=1)  # Concatenate along feature dimension
                    
                
                sequences_all.append(final_seq)
                targets_all.append(ndvi_targets)
                spatial_features_all.append(spatial_features)

        else:
            for i in range(self.grid_size):
                grid_data = self.data[self.data['grid_id'] == i]
                ndvi_series = grid_data['NDVI'].values
                spatial_postion = grid_data['coordinatesArray'].iloc[0][:10]
                sequences, targets, spatial_features = self.create_sliding_windows(ndvi_series, spatial_postion)

                sequences_all.append(sequences)
                targets_all.append(targets)
                spatial_features_all.append(spatial_features)

        # Combine data from all grids
        '''sequences_all = np.vstack(sequences_all)
        targets_all = np.concatenate(targets_all)
        spatial_features_all = np.vstack(spatial_features_all) '''
        
        seq = []
        labelValues = []
        spatial = []
        
        for i in range(np.array(sequences_all).shape[1]):
            for j in range(self.grid_size):
                seq.append(sequences_all[j][i])
                labelValues.append(targets_all[j][i])
                spatial.append(spatial_features_all[j][i])

        scaler = StandardScaler(with_mean=True,with_std=True)
        
        #targetScaler = StandardScaler(with_mean=True,with_std=True)
        print(sequences_all[0].shape)
        sequences_all = torch.tensor(scaler.fit_transform(seq), dtype=torch.float32)
        targets_all = torch.tensor(np.array(labelValues).reshape(-1, 1),dtype=torch.float32)
        spatial_features_all = torch.tensor(scaler.fit_transform(spatial), dtype=torch.float32)
        return sequences_all, spatial_features_all, targets_all

