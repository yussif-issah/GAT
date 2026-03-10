# SPTransformer - Spatial-Temporal Forecasting

A deep learning framework for time series forecasting of vegetation indices using transformer-based and baseline models with grid-based spatial-temporal attention.

## Overview

SPTransformer combines spatial and temporal attention mechanisms to forecast vegetation indices (NDVI, MNDWI, NDMI, NDWI, NBR, EVI, SAVI) from satellite imagery data. The framework implements a custom **Grid Attention Transformer (GAT)** model alongside several baseline models for comparison.

## Features

- 🎯 **Grid Attention Transformer (GAT)**: Custom attention-based model for spatial-temporal forecasting
- 📊 **Multiple Baseline Models**:
  - GRU (Gated Recurrent Unit)
  - LSTM (Long Short-Term Memory)
  - CNN1D (1D Convolutional Neural Network)
  - PatchTST (Patch Time Series Transformer)
  - SimpleInformer
  - TimeSeriesTransformer
- 🔄 **Comprehensive Data Preprocessing**: Handles grid-based satellite data with spatial features
- 📈 **Model Comparison**: Train and test multiple models side-by-side
- 🎮 **Easy Configuration**: Simple parameters for grid size, window size, and forecast horizon

## Project Structure

```
SPTransformer/
├── main.py                          # Main entry point - runs all models
├── models/                          # Model implementations
│   ├── gat.py                       # Grid Attention Transformer
│   └── univariate_baselinemodels.py # Baseline models
├── training/                        # Training framework
│   └── training.py                  # Trainer class for model training/testing
├── data_preprocessing/              # Data handling
│   ├── preprocess.py                # Data preprocessing pipeline
│   ├── dataloaders.py               # PyTorch data loaders
│   └── dataset.py                   # Dataset class
├── data/                            # Input datasets (CSV files)
│   └── indo_indices*.csv            # Vegetation indices data
└── images/                          # Results and outputs
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Installation

1. Clone or download the project:
```bash
cd SPTransformer
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv env
env\Scripts\activate  # On Windows
source env/bin/activate  # On macOS/Linux
```

3. Install dependencies:
```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn
```

## Quick Start

1. **Prepare your data**: Place CSV files containing vegetation indices in the `data/` directory
   - Expected format: CSV with columns for indices (NDVI, MNDWI, etc.) and spatial metadata

2. **Configure parameters** in `main.py`:
```python
DATA_PATH = 'data/indo_indices240_2000_2023.csv'  # Your data file
GRID_SIZE = 240                                    # Grid cells
WINDOW_SIZE = 5                                    # Time window
STEP_SIZE = 1                                      # Sliding window step
FORECAST_HORIZON = 1                               # Forecast steps ahead
```

3. **Run the training**:
```bash
python main.py
```

This will:
- Load and preprocess your data
- Create training/test splits
- Train all models (GAT + 6 baselines)
- Generate test results and visualizations

## Model Details

### Grid Attention Transformer (GAT)
- Spatial attention mechanism for grid cells
- Temporal encoding of NDVI sequences
- Cross-cell attention layers
- LSTM head for final prediction

### Baseline Models
- **GRU/LSTM**: Sequential models with recurrent layers
- **CNN1D**: Convolutional approach to temporal patterns
- **PatchTST**: Patch-based time series transformer
- **SimpleInformer & TimeSeriesTransformer**: Transformer-based baselines

## Configuration

Key parameters in `main.py`:

```python
GRID_SIZE = 240          # Number of grid cells
WINDOW_SIZE = 5          # Time steps in input window
FORECAST_HORIZON = 1     # Steps to forecast ahead
BATCH_SIZE = GRID_SIZE   # Batch size for training
EPOCHS = 50              # Training epochs
LEARNING_RATE = 1e-3     # Optimizer learning rate
```

## Output

Training generates:
- Model performance metrics (MSE loss)
- Attention visualizations (in `attention_logs_*` directories)
- Test results and predictions

## Performance Metrics

Models are evaluated using:
- **MSE (Mean Squared Error)**: Primary loss function
- Test accuracy on held-out data
- Attention map visualizations for interpretability

## Notes

- GPU acceleration is automatically enabled if CUDA is available
- Random seeds are fixed for reproducibility
- Data is normalized using StandardScaler before training
