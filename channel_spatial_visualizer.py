# channel_spatial_visualizer.py

import torch
import matplotlib.pyplot as plt
import numpy as np

# Import everything we need from the main file
from grid_latest_new_explainability import (
    EnhancedTTConvModel,
    test_loader,
    num_channels
    # plus any other objects you need
)

###############################################################################
# 1. Visualize CBAM Channel Attention
###############################################################################
def visualize_cbam_channel_attention(model, X_batch):
    """
    - Run a forward pass to populate CBAM's channel attention.
    - Then plot a bar chart of the average channel attention across the batch.
    """
    device = next(model.parameters()).device
    X_batch = X_batch.to(device)

    model.eval()
    with torch.no_grad():
        _ = model(X_batch)  # This sets model.cbam.last_channel_attention

    # CBAM's channel attention => shape (B, C, 1, 1)
    channel_att = model.cbam.last_channel_attention
    if channel_att is None:
        print("No channel attention found. Make sure the forward pass was done.")
        return
    
    # Average across the batch dimension
    # channel_att_mean => shape (C,)
    channel_att_mean = channel_att.mean(dim=0).squeeze()  # shape => (C,)
    channel_att_mean = channel_att_mean.cpu().numpy()

    # Plot
    plt.figure(figsize=(8,4))
    x_vals = np.arange(len(channel_att_mean))
    plt.bar(x_vals, channel_att_mean, color='orange')
    plt.xlabel("Channel Index")
    plt.ylabel("Attention Weight")
    plt.title("CBAM Channel Attention (Averaged Over Batch)")
    plt.show()

###############################################################################
# 2. Visualize CBAM Spatial Attention
###############################################################################
def visualize_cbam_spatial_attention(model, X_batch):
    """
    - Run a forward pass to populate CBAM's spatial attention.
    - Then visualize the average (or per-sample) map as a heatmap.
    """
    device = next(model.parameters()).device
    X_batch = X_batch.to(device)

    model.eval()
    with torch.no_grad():
        _ = model(X_batch)  # sets model.cbam.last_spatial_attention

    # shape => (B, 1, H, W)
    spatial_att = model.cbam.last_spatial_attention
    if spatial_att is None:
        print("No spatial attention found. Ensure forward pass was done.")
        return

    # Option A: Average across batch => shape (H, W)
    spatial_att_mean = spatial_att.mean(dim=0).squeeze(0).cpu().numpy()

    # Plot
    plt.figure(figsize=(7,6))
    plt.imshow(spatial_att_mean, cmap='hot')
    plt.colorbar()
    plt.title("CBAM Spatial Attention (Averaged Over Batch)")
    plt.show()

    # Option B (Uncomment if you want to see each sample's map)
    # for i in range(spatial_att.shape[0]):
    #     single_map = spatial_att[i, 0].cpu().numpy() # shape (H, W)
    #     plt.figure()
    #     plt.imshow(single_map, cmap='hot')
    #     plt.title(f"Spatial Attention Sample {i}")
    #     plt.colorbar()
    #     plt.show()

###############################################################################
# 3. Example Main
###############################################################################
if __name__ == "__main__":
    # 1) Create model
    model = EnhancedTTConvModel(input_channels=num_channels, num_labels=10)

    # 2) Load best weights
    model.load_state_dict(torch.load('best_model_latest.pth'))
    model.eval()

    # 3) Get a small batch from the test_loader
    sample_data, _ = next(iter(test_loader))
    sample_data = sample_data[:4]  # just the first 4 samples

    # 4) Visualize Channel Attention
    visualize_cbam_channel_attention(model, sample_data)

    # 5) Visualize Spatial Attention
    visualize_cbam_spatial_attention(model, sample_data)
