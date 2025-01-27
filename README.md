# Tensor-encoded_genomic_data


# README

## Overview
This repository contains the implementation of a deep learning model designed for multi-label classification of genomic data using Tensorized Tensor (TT) convolution layers and Convolutional Block Attention Modules (CBAM). The repository is divided into two key scripts:

1. **Main Model Implementation (`main_model.py`)**: Handles data preprocessing, model architecture, training, and evaluation.
2. **Attention Visualizer (`channel_spatial_visualizer.py`)**: Visualizes the attention maps from the CBAM module.

---

## Requirements
The following libraries and frameworks are required to run the code:

- Python 3.8+
- PyTorch==2.0.1
- NumPy==1.24.2
- Pandas==1.3.5
- Matplotlib==3.5.3
- scikit-learn==1.0.2
- imbalanced-learn==0.9.1

Ensure the appropriate versions of CUDA and cuDNN are installed for GPU support.

---

## Files
### `main_model.py`
This script includes the following steps:

1. **Data Preprocessing**:
   - Filters valid genomic variants based on base substitutions.
   - Maps variant classifications and base substitutions to one-hot vectors.
   - Constructs a 4D tensor representing samples with genomic features.
   - Handles class imbalance using `RandomOverSampler`.

2. **Model Architecture**:
   - Incorporates CBAM for channel and spatial attention.
   - Utilizes TT convolution layers for efficient parameterization.
   - Combines these layers with global pooling and dense layers for final classification.

3. **Training and Evaluation**:
   - Trains the model using `AdamW` optimizer with a learning rate scheduler.
   - Implements early stopping based on validation loss.
   - Saves the best model weights to `best_model_latest.pth`.

4. **Outputs**:
   - Displays training and validation performance metrics.
   - Saves the trained model for inference or visualization.

### `channel_spatial_visualizer.py`
This script visualizes the learned attention maps from the CBAM module:

1. **Channel Attention Visualization**:
   - Displays a bar chart showing the average attention weights across all channels in the batch.

2. **Spatial Attention Visualization**:
   - Generates a heatmap representing the average spatial attention map.

---

## How to Run

### Main Model
1. Place your data file (`integrated_data_filtered_distributed_labels.csv`) in the repository.
2. Update the file path in `main_model.py`.
3. Run the script:
   ```bash
   python main_model.py
   ```
4. The trained model weights will be saved as `best_model_latest.pth`.

### Attention Visualization
1. Ensure `best_model_latest.pth` is present in the repository.
2. Run the visualizer script:
   ```bash
   python channel_spatial_visualizer.py
   ```
3. View the generated attention visualizations.

---

## Key Features
1. **CBAM Integration**:
   - Enhances the model’s focus on informative features with channel and spatial attention mechanisms.
2. **TT Convolution Layers**:
   - Reduces memory overhead and enhances efficiency by factorizing convolutional weights.
3. **Explainability Tools**:
   - Visualize how the model pays attention to different channels and spatial regions.

---

## Results
- The trained model achieves effective multi-label classification of genomic data.
- Attention visualizations provide insights into the model's decision-making process.












![architecture_new](https://github.com/user-attachments/assets/f7326905-a1c9-4cc8-b040-1d4c91b63070)
