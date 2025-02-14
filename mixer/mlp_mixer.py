import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, csv_file, n_bins=1024):
        df = pd.read_csv(csv_file)
        
        # Separate input and output
        X = df.iloc[:, :16].values
        y = df.iloc[:, 16:].values
        
        # Custom binning function
        self.bin_edges_X = []
        self.X_binned = np.zeros_like(X, dtype=np.int64)
        for i in range(X.shape[1]):
            quantiles = np.linspace(0, 100, n_bins + 1)
            edges = np.percentile(X[:, i], quantiles)
            edges[-1] = edges[-1] + 1e-8  # Ensure the last value gets included
            self.bin_edges_X.append(edges)
            self.X_binned[:, i] = np.digitize(X[:, i], edges[1:-1])
        
        self.bin_edges_y = []
        self.y_binned = np.zeros_like(y, dtype=np.int64)
        for i in range(y.shape[1]):
            quantiles = np.linspace(0, 100, n_bins + 1)
            edges = np.percentile(y[:, i], quantiles)
            edges[-1] = edges[-1] + 1e-8  # Ensure the last value gets included
            self.bin_edges_y.append(edges)
            self.y_binned[:, i] = np.digitize(y[:, i], edges[1:-1])
        
        self.X = torch.tensor(self.X_binned, dtype=torch.long)
        self.y = torch.tensor(self.y_binned, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def inverse_transform_y(self, y_binned):
        result = np.zeros_like(y_binned, dtype=np.float32)
        for i in range(y_binned.shape[1]):
            edges = self.bin_edges_y[i]
            bin_centers = (edges[1:] + edges[:-1]) / 2
            result[:, i] = bin_centers[y_binned[:, i]]
        return result

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class MLPMixer(nn.Module):
    def __init__(self, 
                 n_tokens=1024,
                 n_channels=16,
                 token_dim=256):
        super().__init__()
        
        # Embedding table for input tokens
        self.embedding = nn.Embedding(n_tokens, token_dim)
        
        # Token-mixing (across channels)
        self.token_mix = nn.Linear(n_channels, n_channels, bias=False)
        
        # Channel-mixing (across token dimension)
        self.channel_mix = nn.Linear(token_dim, token_dim, bias=False)
        
        # Output head
        self.output_head = nn.Linear(token_dim, n_tokens, bias=False)
        
        self.act = Swish()
        
    def forward(self, x):
        # Embed the tokens
        x = self.embedding(x)  # shape: (batch_size, n_channels, token_dim)
        
        # Token mixing
        x = x.permute(0, 2, 1)  # (batch_size, token_dim, n_channels)
        x = self.token_mix(x)  # Mix across channels
        x = self.act(x)
        x = x.permute(0, 2, 1)  # (batch_size, n_channels, token_dim)
        
        # Channel mixing
        x = self.channel_mix(x)  # Mix across token dimension
        x = self.act(x)
        
        # Output predictions
        x = self.output_head(x)  # shape: (batch_size, n_channels, n_tokens)
        
        return x

def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_X)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = batch_y.view(-1)
            
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

def compare_predictions(model, dataset, device, num_samples=5):
    model.eval()
    X = dataset.X[:num_samples].to(device)
    
    with torch.no_grad():
        outputs = model(X)
        predictions = outputs.argmax(dim=-1).cpu().numpy()
        
        continuous_preds = dataset.inverse_transform_y(predictions)
        true_values = dataset.inverse_transform_y(dataset.y[:num_samples].numpy())
        
        print("\nPredictions vs True Values:")
        print("-" * 80)
        for i in range(num_samples):
            print(f"\nSample {i+1}:")
            print("Predicted:", ", ".join([f"{x:.3f}" for x in continuous_preds[i]]))
            print("True:     ", ", ".join([f"{x:.3f}" for x in true_values[i]]))
            
            mse = np.mean((continuous_preds[i] - true_values[i])**2)
            print(f"MSE: {mse:.6f}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    n_bins = 1024
    token_dim = 256
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 30

    # Load dataset
    dataset = CustomDataset('20250214_091646_data.csv', n_bins=n_bins)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = MLPMixer(
        n_tokens=n_bins,
        n_channels=16,
        token_dim=token_dim
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs, device)

    # Compare predictions with true values
    compare_predictions(model, dataset, device)

    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'bin_edges_X': dataset.bin_edges_X,
        'bin_edges_y': dataset.bin_edges_y
    }, 'mlp_mixer.pth')

if __name__ == "__main__":
    main()