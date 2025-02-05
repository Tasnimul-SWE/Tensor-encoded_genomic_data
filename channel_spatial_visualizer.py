import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable

# === 1) Import from your main file ===
from today_code import (
    EnhancedTTConvModel,
    test_loader,
    variant_classification_mapping,
    base_substitution_mapping
)s

def make_channel_names():
    """Create ordered channel names matching model's input structure."""
    variant_class_names = sorted(
        variant_classification_mapping,
        key=variant_classification_mapping.get
    )
    base_subst_names = sorted(
        base_substitution_mapping,
        key=base_substitution_mapping.get
    )
    return variant_class_names + base_subst_names

def visualize_synthetic_spatial_attention_sparse(data_loader, num_samples=16):
    """Visualize attention map with color gradient and clear cell borders."""
    height, width = 310, 313
    np.random.seed(42)

    # Generate synthetic attention with varying intensities
    synthetic_attention = np.random.rand(height, width) * 0.2  # Base noise

    # Increase the number of high-attention cells
    num_high = 1000  # Increased from 150 to 1000
    high_positions = np.random.choice(height * width, num_high, replace=False)
    for idx in high_positions:
        i, j = idx // width, idx % width
        synthetic_attention[i, j] = 1  # Set individual cells to high attention

    # Plot with color gradient
    fig, ax = plt.subplots(figsize=(16, 12))
    cmap = plt.cm.coolwarm  # Gradient from cool (low attention) to warm (high attention)
    norm = Normalize(vmin=0, vmax=1)  # Normalize attention values between 0 and 1

    attention_img = ax.imshow(synthetic_attention, cmap=cmap, norm=norm, interpolation='none', aspect='auto')

    # Configure grid to match cell boundaries with distinct borders
    ax.set_xticks(np.arange(-0.5, width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, height, 1), minor=True)
    ax.grid(which='minor', color='black', linewidth=0.5, linestyle='-')  # Dark gridlines for clear cell borders

    # Add color bar to interpret attention values
    cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label('Attention Intensity')

    plt.title("Attention Map Visualization with Clear Cell Borders")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    channel_names = make_channel_names()

    model = EnhancedTTConvModel(input_channels=25, num_labels=10)
    model.load_state_dict(torch.load('today.pth'))
    model.eval()

    visualize_synthetic_spatial_attention_sparse(test_loader)
