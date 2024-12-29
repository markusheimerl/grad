import pandas as pd
import numpy as np
import jax.numpy as jnp
from sklearn.preprocessing import StandardScaler
import pickle
import os

class ControlDataset:
    def __init__(self, file_path, sequence_length=256, scale_data=True):
        """
        Initialize the Control Dataset
        
        Args:
            file_path (str): Path to the CSV file containing control data
            sequence_length (int): Length of sequences to generate
            scale_data (bool): Whether to standardize the features
        """
        # Load CSV
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        
        # Extract the columns we want to predict
        self.target_columns = [
            'ang_vel[0]', 'ang_vel[1]', 'ang_vel[2]',
            'acc[0]', 'acc[1]', 'acc[2]',
            'omega[0]', 'omega[1]', 'omega[2]', 'omega[3]'
        ]
        
        # Extract features
        self.data = df[self.target_columns].values
        self.sequence_length = sequence_length
        self.feature_dim = len(self.target_columns)
        
        # Scale the data if requested
        self.scaler = None
        if scale_data:
            scaler_path = file_path + '.scaler'
            if os.path.exists(scaler_path):
                print("Loading existing scaler...")
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.data = self.scaler.transform(self.data)
            else:
                print("Creating and fitting new scaler...")
                self.scaler = StandardScaler()
                self.data = self.scaler.fit_transform(self.data)
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
        
        print(f"Dataset initialized with {len(self.data)} samples")
        print(f"Feature dimension: {self.feature_dim}")
        print(f"Sequence length: {self.sequence_length}")
        print(f"Number of sequences: {len(self)}")

    def __len__(self):
        """Return the number of possible sequences"""
        return (len(self.data) - (self.sequence_length + 1)) // self.sequence_length

    def __getitem__(self, idx):
        """Get a sequence and its corresponding next-step values"""
        idx = idx * self.sequence_length
        # Get sequence and next sequence as labels
        inputs = jnp.array(self.data[idx : idx + self.sequence_length], dtype=jnp.float32)
        labels = jnp.array(self.data[idx + 1 : idx + self.sequence_length + 1], dtype=jnp.float32)
        return inputs, labels

    def inverse_transform(self, scaled_data):
        """Convert scaled data back to original scale"""
        if self.scaler is not None:
            return self.scaler.inverse_transform(scaled_data)
        return scaled_data

    def get_feature_stats(self):
        """Return feature means and standard deviations"""
        if self.scaler is not None:
            return {
                'mean': self.scaler.mean_,
                'scale': self.scaler.scale_
            }
        return {
            'mean': np.zeros(self.feature_dim),
            'scale': np.ones(self.feature_dim)
        }

def create_train_val_datasets(file_path, sequence_length=256, val_split=0.1, scale_data=True):
    """
    Create training and validation datasets
    
    Args:
        file_path (str): Path to the CSV file
        sequence_length (int): Length of sequences
        val_split (float): Fraction of data to use for validation
        scale_data (bool): Whether to standardize the features
    
    Returns:
        tuple: (training dataset, validation dataset)
    """
    # Load full dataset
    full_dataset = ControlDataset(file_path, sequence_length, scale_data)
    
    # Calculate split point
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    
    # Split the data
    train_data = full_dataset.data[:train_size * sequence_length]
    val_data = full_dataset.data[train_size * sequence_length:]
    
    # Create separate dataset objects
    train_dataset = ControlDataset(file_path, sequence_length, False)  # False because data is already scaled
    train_dataset.data = train_data
    train_dataset.scaler = full_dataset.scaler
    
    val_dataset = ControlDataset(file_path, sequence_length, False)
    val_dataset.data = val_data
    val_dataset.scaler = full_dataset.scaler
    
    print(f"\nTraining sequences: {len(train_dataset)}")
    print(f"Validation sequences: {len(val_dataset)}")
    
    return train_dataset, val_dataset

if __name__ == "__main__":
    # Example usage
    dataset = ControlDataset("2024-12-29_6-25-1_control_data.csv", sequence_length=256)
    
    # Get a sample batch
    inputs, labels = dataset[0]
    print("\nSample batch shapes:")
    print(f"Inputs shape: {inputs.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Create train/val split
    train_dataset, val_dataset = create_train_val_datasets(
        "2024-12-29_6-25-1_control_data.csv",
        sequence_length=256,
        val_split=0.1
    )
