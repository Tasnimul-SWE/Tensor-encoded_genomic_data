import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

###############################################################################
# 1. Read and preprocess the data
###############################################################################
file_path = 'integrated_data_filtered_distributed_labels.csv'
data = pd.read_csv(file_path)

# Valid base substitutions
valid_base_substitutions = [
    'C->T', 'C->A', 'G->A', 'A->C', 'T->A', 'G->T',
    'G->C', 'T->G', 'A->G', 'T->C', 'A->T', 'C->G'
]
data['Base_Substitution'] = data['Reference_Allele'] + '->' + data['Tumor_Seq_Allele2']

# Filter rows to keep only valid base substitutions
filtered_data = data[data['Base_Substitution'].isin(valid_base_substitutions)].reset_index(drop=True)

# Mappings for Variant Classification and Base Substitution
variant_classification_categories = filtered_data['Variant_Classification'].unique()
variant_classification_mapping = {cat: idx for idx, cat in enumerate(variant_classification_categories)}
base_substitution_mapping = {sub: idx for idx, sub in enumerate(valid_base_substitutions)}

num_variant_classes = len(variant_classification_mapping)
num_base_subst = len(base_substitution_mapping)

# Build the 4D array
height, width = 310, 313
num_channels = num_variant_classes + num_base_subst

unique_samples = filtered_data['Tumor_Sample_Barcode'].unique()
num_samples = len(unique_samples)

cnn_data = np.zeros((num_samples, height, width, num_channels), dtype=np.float32)
for sample_idx, sample_id in enumerate(unique_samples):
    sample_data = filtered_data[filtered_data['Tumor_Sample_Barcode'] == sample_id]
    sample_grid = np.zeros((height, width, num_channels), dtype=np.float32)
    for i, row in sample_data.iterrows():
        variant_idx = (i - sample_data.index[0])
        grid_row = variant_idx // width
        grid_col = variant_idx % width
        if grid_row >= height:
            break

        # One-hot for variant classification
        vc_vector = np.zeros(num_variant_classes, dtype=np.float32)
        vc_vector[variant_classification_mapping[row['Variant_Classification']]] = 1.0

        # One-hot for base substitution
        bs_vector = np.zeros(num_base_subst, dtype=np.float32)
        bs_vector[base_substitution_mapping[row['Base_Substitution']]] = 1.0

        full_channel = np.concatenate([vc_vector, bs_vector])
        sample_grid[grid_row, grid_col, :] = full_channel
    
    cnn_data[sample_idx] = sample_grid

# Prepare labels (exactly one label per sample)
label_columns = [
    'Aggressive_Prostate Cancer', 'Indolent_Prostate Cancer',
    'Aggressive_Breast Cancer', 'Indolent_Breast Cancer',
    'Aggressive_Ovarian Cancer', 'Indolent_Ovarian Cancer',
    'Aggressive_Pancreatic Cancer', 'Indolent_Pancreatic Cancer',
    'Aggressive_Colorectal Cancer', 'Indolent_Colorectal Cancer'
]
filtered_labels = filtered_data.groupby('Tumor_Sample_Barcode')[label_columns].first()
assert (filtered_labels.sum(axis=1) == 1).all(), "Each sample must belong to exactly one label!"
labels_int = filtered_labels.values.argmax(axis=1)

# Handle class imbalance with RandomOverSampler
X_flat = cnn_data.reshape(cnn_data.shape[0], -1)
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_flat, labels_int)

# Reshape back to (N, H, W, C)
cnn_data = X_resampled.reshape(-1, height, width, num_channels)

# Normalize to [0, 1]
cnn_data_min = cnn_data.min()
cnn_data_max = cnn_data.max()
cnn_data = (cnn_data - cnn_data_min) / (cnn_data_max - cnn_data_min + 1e-7)

labels_tensor = torch.tensor(y_resampled, dtype=torch.long)

# Split train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    cnn_data, labels_tensor, test_size=0.2, random_state=42
)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

###############################################################################
# 2. Define CBAM Module (with saved attention maps)
###############################################################################
class CBAM(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        # Channel attention
        self.channel_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(input_channels, input_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(input_channels // reduction_ratio, input_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
        # Spatial attention
        self.conv = nn.Conv2d(
            in_channels=2, out_channels=1,
            kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )

        # -- To store attention
        self.last_channel_attention = None
        self.last_spatial_attention = None

    def forward(self, x):
        b, c, h, w = x.size()
        
        # Channel Attention
        avg_out = self.shared_mlp(self.channel_avg_pool(x).view(b, c)).view(b, c, 1, 1)
        max_out = self.shared_mlp(self.channel_max_pool(x).view(b, c)).view(b, c, 1, 1)
        channel_attention = self.sigmoid(avg_out + max_out)
        # Save for analysis
        self.last_channel_attention = channel_attention.detach().cpu().clone()
        x = x * channel_attention
        
        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)   # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True) # (B, 1, H, W)
        spatial_attention = self.sigmoid(
            self.conv(torch.cat([avg_out, max_out], dim=1))
        )
        # Save for analysis
        self.last_spatial_attention = spatial_attention.detach().cpu().clone()

        x = x * spatial_attention
        return x

###############################################################################
# 3. Define a pseudo "TTConv" layer (actually uses standard conv + mixing)
###############################################################################
class TTConvLayer(nn.Module):
    def __init__(self, window, inp_ch_modes, out_ch_modes, ranks, stride=1, padding='same'):
        super(TTConvLayer, self).__init__()
        
        self.total_inp_ch = np.prod(inp_ch_modes)  # e.g. 25
        self.total_out_ch = np.prod(out_ch_modes)  # e.g. 25
        
        # 2D conv
        self.conv = nn.Conv2d(
            in_channels=self.total_inp_ch,
            out_channels=self.total_out_ch,
            kernel_size=window,
            stride=stride,
            padding=padding
        )
        self.bn = nn.BatchNorm2d(self.total_out_ch)
        
        # 1D channel mixing (pretend these are TT "cores")
        self.channel_mix = nn.Sequential(
            nn.Conv1d(self.total_out_ch, self.total_out_ch, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(self.total_out_ch, self.total_out_ch, kernel_size=1)
        )

        # We'll store "core attention" for demonstration
        self.core_attention_map = None

    def forward(self, x):
        # x shape: (B, H, W, C)
        bsz, h, w, c = x.shape
        x = x.permute(0, 3, 1, 2)  # => (B, C, H, W)
        
        x = self.conv(x)
        x = self.bn(x)
        
        bsz, out_c, h_out, w_out = x.shape
        
        # Flatten the spatial dims => (B, out_c, H_out*W_out)
        x = x.reshape(bsz, out_c, -1)
        x = self.channel_mix(x)
        
        # Approximate "core importance" by mean activation across channels
        with torch.no_grad():
            core_att = x.mean(dim=-1, keepdim=True)  # => (B, out_c, 1)
            core_att = core_att.mean(dim=0, keepdim=True)  # => (1, out_c, 1)
            self.core_attention_map = core_att.detach().cpu().clone()

        # Reshape back to (B, out_c, H_out, W_out)
        x = x.reshape(bsz, out_c, h_out, w_out)
        x = x.permute(0, 2, 3, 1)  # => (B, H_out, W_out, out_c)
        return x

###############################################################################
# 4. Define the Enhanced TTConv Model with attention merging
###############################################################################
class EnhancedTTConvModel(nn.Module):
    def __init__(self, input_channels=25, num_labels=10):
        super(EnhancedTTConvModel, self).__init__()
        
        #######################################################################
        # 1) CBAM
        #######################################################################
        self.cbam = CBAM(input_channels)
        
        #######################################################################
        # 2) Two TTConv layers
        #######################################################################
        self.tt_conv1 = TTConvLayer(
            window=[3, 3],
            inp_ch_modes=[5, 5],
            out_ch_modes=[5, 5],
            ranks=[4, 4]
        )
        self.tt_conv2 = TTConvLayer(
            window=[3, 3],
            inp_ch_modes=[5, 5],
            out_ch_modes=[5, 5],
            ranks=[4, 4]
        )
        
        #######################################################################
        # 3) Global Max Pool & final MLP
        #######################################################################
        self.global_pool = nn.AdaptiveMaxPool2d((1,1))
        self.final_layers = nn.Sequential(
            nn.Linear(25, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_labels)
        )
        
        self.dropout = nn.Dropout(0.3)
        
        # We'll store TTConv attentions for analysis
        self.tt_att_maps = []

    def forward(self, x):
        # x shape: (B, H, W, C)
        
        # === 1) CBAM ===
        x = x.permute(0, 3, 1, 2)  # => (B, C, H, W)
        x = self.cbam(x)
        
        # === TTConv 1 ===
        x = x.permute(0, 2, 3, 1)  # => (B, H, W, C)
        x = self.tt_conv1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # === TTConv 2 ===
        x = self.tt_conv2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Store the "core attention" from each TTConv
        self.tt_att_maps = [
            self.tt_conv1.core_attention_map,
            self.tt_conv2.core_attention_map
        ]
        
        # === Global Max Pool ===
        x = x.permute(0, 3, 1, 2)  # => (B, 25, H, W)
        x = self.global_pool(x)    # => (B, 25, 1, 1)
        x = x.view(x.size(0), -1)  # => (B, 25)
        
        # === Final MLP ===
        x = self.final_layers(x)
        return x

    def get_merged_attention(self, merge_mode='sum'):
        """
        Merge CBAM's spatial attention with the TTConv "core" attention to produce a single map.
        We'll do a simplified approach:
          - average across batch dimension for the CBAM spatial map,
          - sum or average the TTConv "core" attentions,
          - combine them with a sum or product.
        """
        # 1) Retrieve CBAM's last spatial attention
        cbam_spatial = self.cbam.last_spatial_attention  # shape: (B, 1, H, W)
        if cbam_spatial is None or len(self.tt_att_maps) == 0:
            print("No attention maps available. Run a forward pass first.")
            return None
        
        # average across batch
        cbam_spatial = cbam_spatial.mean(dim=0)  # => (1, H, W)
        
        # 2) Merge TTConv attention
        merged_tt_core = None
        for att_map in self.tt_att_maps:
            if att_map is not None:
                # shape: (1, out_c, 1)
                if merged_tt_core is None:
                    merged_tt_core = att_map
                else:
                    merged_tt_core += att_map
        if merged_tt_core is None:
            print("TTConv core maps are None.")
            return cbam_spatial.squeeze(0).numpy()
        
        # We reduce out_c dimension -> scalar
        # e.g., average across channels
        alpha = merged_tt_core.mean(dim=1)  # => shape (1,1)
        alpha_value = alpha.item()
        
        # 3) Combine the two
        if merge_mode == 'sum':
            merged_map = cbam_spatial + alpha_value
        else:
            # 'product'
            merged_map = cbam_spatial * alpha_value
        
        return merged_map.squeeze(0).detach().cpu().numpy()

###############################################################################
# 5. Train & evaluate
###############################################################################
def train_model(model, train_loader, test_loader, num_epochs=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    patience = 50
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            correct_train += (predicted == y_batch).sum().item()
            total_train += y_batch.size(0)

        train_acc = correct_train / total_train
        
        # Validation
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for X_val, y_val in test_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs_val = model(X_val)
                loss_val = loss_fn(outputs_val, y_val)
                val_loss += loss_val.item()
                _, predicted_val = torch.max(outputs_val, dim=1)
                correct_val += (predicted_val == y_val).sum().item()
                total_val += y_val.size(0)

        val_acc = correct_val / total_val
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} "
              f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'today.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

def test_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    
    test_accuracy = correct / total
    print(f"Test Accuracy: {test_accuracy:.4f}")
    return test_accuracy

###############################################################################
# 6. Visualization of merged attention
###############################################################################
def visualize_merged_attention(model, sample_batch):
    """
    1) Forward pass -> generate CBAM + TTConv attention.
    2) Merge them -> single 2D map (310 x 313).
    3) Plot the map.
    """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sample_batch = sample_batch.to(device)

    with torch.no_grad():
        _ = model(sample_batch)  # forward pass to populate attention maps

    merged_map = model.get_merged_attention(merge_mode='sum')  # or 'product'
    if merged_map is None:
        print("Merged attention is None.")
        return

    # Expect shape ~ (310, 313) if everything lines up
    plt.figure(figsize=(7,6))
    plt.imshow(merged_map, cmap='hot')
    plt.colorbar()
    plt.title("Merged Attention (CBAM + TTConv)")
    plt.show()


###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    model = EnhancedTTConvModel(input_channels=num_channels, num_labels=10)
    train_model(model, train_loader, test_loader, num_epochs=200)
    
    # Load best weights
    model.load_state_dict(torch.load('today.pth'))
    test_model(model, test_loader)

    # Visualize attention on a small sample batch
    sample_data, _ = next(iter(test_loader))  # get a single batch
    sample_data = sample_data[:4]            # pick first 4 samples for demonstration
    visualize_merged_attention(model, sample_data)