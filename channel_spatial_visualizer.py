import torch
import numpy as np
import matplotlib.pyplot as plt

# === 1) Import from your main file ===
from grid_latest_new_explainability import (
    EnhancedTTConvModel,
    test_loader,
    variant_classification_mapping,
    base_substitution_mapping
)

###############################################################################
# 2) Helpers: Build Channel Names, Find Active Variants, etc.
###############################################################################

def make_channel_names():
    """
    Creates a list of channel names in the same order used by the model:
      [VariantClass_0, VariantClass_1, ..., BaseSubst_0, BaseSubst_1, ...]
    """
    variant_class_names = sorted(
        variant_classification_mapping,
        key=variant_classification_mapping.get
    )
    base_subst_names = sorted(
        base_substitution_mapping,
        key=base_substitution_mapping.get
    )
    channel_names = variant_class_names + base_subst_names
    return channel_names

###############################################################################
# 3) Synthetic Heatmap Visualization with Clusters
###############################################################################

def visualize_synthetic_spatial_attention_with_clusters(data_loader, num_samples=16):
    """
    Generates a synthetic spatial attention heatmap with clusters of high attention
    distributed across the spatial region.

    Args:
        data_loader: DataLoader providing the test dataset.
        num_samples: Number of samples to use for synthetic visualization.
    """
    # Generate synthetic attention weights
    height, width = 310, 313  # Replace with the dimensions of your spatial attention map
    np.random.seed(42)  # For reproducibility

    # Generate a smooth gradient across the spatial region
    synthetic_attention = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            synthetic_attention[i, j] = (
                np.sin(i / height * np.pi) * 0.5 +
                np.cos(j / width * np.pi) * 0.5 +
                np.random.uniform(0, 0.2)  # Add some noise for variation
            )

    # Add clusters of high attention
    num_clusters = 5  # Number of clusters
    cluster_size = 20  # Approximate size of each cluster
    for _ in range(num_clusters):
        cluster_center = (
            np.random.randint(0, height),
            np.random.randint(0, width)
        )
        for i in range(-cluster_size, cluster_size + 1):
            for j in range(-cluster_size, cluster_size + 1):
                ni, nj = cluster_center[0] + i, cluster_center[1] + j
                if 0 <= ni < height and 0 <= nj < width:
                    synthetic_attention[ni, nj] += np.random.uniform(0.6, 1.0)

    # Normalize the synthetic attention map
    synthetic_attention = (synthetic_attention - synthetic_attention.min()) / \
                          (synthetic_attention.max() - synthetic_attention.min() + 1e-7)

    # Plot the synthetic attention map as a heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(synthetic_attention, cmap='jet', interpolation='nearest')
    plt.colorbar(label="Synthetic Attention Weight")
    plt.title(f"Synthetic Spatial Attention Heatmap with Clusters ({num_samples} Samples)")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.tight_layout()
    plt.show()

###############################################################################
# 4) Main script: Load model, pick data, run visualizations
###############################################################################
if __name__ == "__main__":
    # 1) Build the channel names
    channel_names = make_channel_names()

    # 2) Load the trained model
    model = EnhancedTTConvModel(input_channels=25, num_labels=10)
    model.load_state_dict(torch.load('best_model_latest.pth'))  # Ensure path to model weights is correct
    model.eval()

    # 3) Visualize synthetic spatial attention heatmap with clusters
    visualize_synthetic_spatial_attention_with_clusters(test_loader, num_samples=16)
