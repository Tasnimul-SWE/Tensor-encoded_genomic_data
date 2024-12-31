# %%
import pandas as pd
import numpy as np
import torch

# %%
file_path = 'integrated_data_filtered_distributed_labels.csv'
data = pd.read_csv(file_path)

# %%
# Define valid base substitutions
valid_base_substitutions = [
    'C->T', 'C->A', 'G->A', 'A->C', 'T->A', 'G->T',
    'G->C', 'T->G', 'A->G', 'T->C', 'A->T', 'C->G'
]


# %%
data['Base_Substitution'] = data['Reference_Allele'] + '->' + data['Tumor_Seq_Allele2']


# %%
filtered_data = data[data['Base_Substitution'].isin(valid_base_substitutions)].reset_index(drop=True)


# %%
# Check the number of variants in the filtered data
num_variants_filtered = filtered_data['Variant'].nunique()
num_variants_filtered


# %%
variant_classification_categories = filtered_data['Variant_Classification'].unique()
variant_classification_mapping = {cat: idx for idx, cat in enumerate(variant_classification_categories)}

# %%
num_categories = len(variant_classification_categories)
print(f"Number of unique Variant_Classification categories: {num_categories}")

# %%
print("Variant Classification Mapping:")
for classification, idx in variant_classification_mapping.items():
    print(f"{classification}: {idx}")


# %%
base_substitution_mapping = {sub: idx for idx, sub in enumerate(valid_base_substitutions)}


# %%
print("Base Substitution Mapping:")
for substitution, idx in base_substitution_mapping.items():
    print(f"{substitution}: {idx}")

# %%
height, width = 310, 313
num_channels = len(variant_classification_mapping) + len(base_substitution_mapping) 

# %%
unique_samples = filtered_data['Tumor_Sample_Barcode'].unique()
num_samples = len(unique_samples)

# %%
print(num_samples)

# %%
print(unique_samples)

# %%
cnn_data = np.zeros((num_samples, height, width, num_channels), dtype=np.float32)


# %%
for sample_idx, sample_id in enumerate(unique_samples):
    sample_data = filtered_data[filtered_data['Tumor_Sample_Barcode'] == sample_id]
    
    # Initialize grid for this sample
    sample_grid = np.zeros((height, width, num_channels), dtype=np.float32)
    
    for variant_idx, row in sample_data.iterrows():
        # Calculate grid position
        grid_row = variant_idx // width
        grid_col = variant_idx % width
        
        if grid_row >= height:
            break
        
        # Create one-hot encoding for Variant Classification
        variant_classification_vector = np.zeros(len(variant_classification_mapping), dtype=np.float32)
        if row['Variant_Classification'] in variant_classification_mapping:
            variant_classification_vector[variant_classification_mapping[row['Variant_Classification']]] = 1
        
        # Create one-hot encoding for Base Substitution
        base_substitution_vector = np.zeros(len(base_substitution_mapping), dtype=np.float32)
        if row['Base_Substitution'] in base_substitution_mapping:
            base_substitution_vector[base_substitution_mapping[row['Base_Substitution']]] = 1
        
        # Combine both into a full channel representation
        full_channel_representation = np.concatenate([variant_classification_vector, base_substitution_vector])
        
        # Assign the representation to the grid
        sample_grid[grid_row, grid_col, :] = full_channel_representation
    
    # Add the grid for this sample to the overall data tensor
    cnn_data[sample_idx] = sample_grid

# %%
input_tensor = torch.tensor(cnn_data, dtype=torch.float32)


# %%
print(f"Input tensor shape: {input_tensor.shape}")  # Expected shape: (521, 310, 313, 25)


# %%
# Print a portion of the grid for a specific sample
sample_index = 0  # Choose the first sample as an example
print(f"Sample ID: {unique_samples[sample_index]}")
print("Grid (slice):", cnn_data[sample_index, :5, :5, :])  # Print the first 5x5 cells and their channels



# %%
print(filtered_data.iloc[:, -10:].dtypes)

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# %%
print(filtered_data.iloc[:, -10:].dtypes)


# %%
# Extract labels from the last 10 columns of the filtered data
label_columns = filtered_data.iloc[:, -10:]  # Adjust this based on the actual label location
labels_numeric = label_columns.apply(pd.to_numeric, errors='coerce').fillna(0)  # Convert to numeric and handle missing values

# Convert to PyTorch tensor
labels = torch.tensor(labels_numeric.values, dtype=torch.float32)

# Print shape to confirm
print(f"Labels shape: {labels.shape}")  # Should match the number of samples in filtered_data


# %%
# Assuming the filtered_data from the preprocessing steps is already available.
# If filtered_data is not available, it needs to be defined from earlier preprocessing steps.

# Define the label columns
label_columns = ['Aggressive_Prostate Cancer', 'Indolent_Prostate Cancer', 'Aggressive_Breast Cancer',
                 'Indolent_Breast Cancer', 'Aggressive_Ovarian Cancer', 'Indolent_Ovarian Cancer',
                 'Aggressive_Pancreatic Cancer', 'Indolent_Pancreatic Cancer',
                 'Aggressive_Colorectal Cancer', 'Indolent_Colorectal Cancer']

# Group the filtered data by 'Tumor_Sample_Barcode' and take the first occurrence for each label
filtered_labels = filtered_data.groupby('Tumor_Sample_Barcode')[label_columns].first()

# Verify that each sample belongs to exactly one label
assert (filtered_labels.sum(axis=1) == 1).all(), "Each sample must belong to exactly one label!"

# Convert the grouped labels to a PyTorch tensor
import torch
labels_tensor = torch.tensor(filtered_labels.values, dtype=torch.float32)

# Display the shape and the first few rows of the labels tensor
filtered_labels_shape = labels_tensor.shape
filtered_labels_head = labels_tensor[:5]

filtered_labels_shape, filtered_labels_head


# %%
# Get the index of the sample in the grouped filtered data
sample_name = "Aggressive_Pancreatic Cancer_4"
sample_index = filtered_labels.index.get_loc(sample_name)

# Retrieve the one-hot encoded label for the sample
one_hot_label = labels_tensor[sample_index]

# Match the one-hot encoded label to the cancer group
cancer_group = label_columns[torch.argmax(one_hot_label).item()]

print(f"Sample {sample_name} belongs to the cancer group: {cancer_group}")


# %%
# Define the sample name
sample_name = "Aggressive_Prostate Cancer_5"

# Get the index of the sample in the grouped filtered data
sample_index = filtered_labels.index.get_loc(sample_name)

# Retrieve the one-hot encoded label for the sample
one_hot_label = labels_tensor[sample_index]

# Display the one-hot encoded label
print(f"One-hot encoded label for {sample_name}: {one_hot_label}")






