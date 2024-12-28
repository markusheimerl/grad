import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from time import time

class DroneDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DroneNN(nn.Module):
    def __init__(self, input_size):
        super(DroneNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 4)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

def clean_data(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df[df.abs().max(axis=1) < 1e6]
    return df

def evaluate_model(model, data_loader, criterion, scaler=None):
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X, y in data_loader:
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            
            if scaler:
                predictions.extend(scaler.inverse_transform(outputs.numpy()))
                actuals.extend(scaler.inverse_transform(y.numpy()))
            
    avg_loss = total_loss / len(data_loader)
    
    if scaler:
        mse = np.mean((np.array(predictions) - np.array(actuals)) ** 2)
        rmse = np.sqrt(mse)
        return avg_loss, rmse
    
    return avg_loss

def plot_learning_curves(train_losses, val_losses, rmse_values):
    plt.figure(figsize=(12, 4))
    
    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Learning Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot RMSE
    plt.subplot(1, 2, 2)
    plt.plot(rmse_values, label='RMSE')
    plt.title('Root Mean Square Error')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.close()

def train_drone_model(csv_file, epochs=100, batch_size=32, early_stopping_patience=10):
    # Load and prepare data
    print("Loading and preparing data...")
    df = pd.read_csv(csv_file)
    print("Original shape:", df.shape)
    df = clean_data(df)
    print("Shape after cleaning:", df.shape)
    
    feature_columns = ['linear_position_d_W[0]', 'linear_position_d_W[1]', 'linear_position_d_W[2]',
                      'yaw_d', 'angular_velocity_B[0]', 'angular_velocity_B[1]', 'angular_velocity_B[2]',
                      'linear_acceleration_B[0]', 'linear_acceleration_B[1]', 'linear_acceleration_B[2]']
    
    target_columns = ['omega_next[0]', 'omega_next[1]', 'omega_next[2]', 'omega_next[3]']
    
    X = df[feature_columns].values
    y = df[target_columns].values
    
    # Scale data
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)
    
    # Split data into train, validation, and test sets
    X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
    
    # Create data loaders
    train_dataset = DroneDataset(X_train, y_train)
    val_dataset = DroneDataset(X_val, y_val)
    test_dataset = DroneDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model, loss function, and optimizer
    model = DroneNN(input_size=X.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training tracking variables
    train_losses = []
    val_losses = []
    rmse_values = []
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time()
    
    print("\nStarting training...")
    print(f"{'Epoch':>5} {'Train Loss':>12} {'Val Loss':>12} {'RMSE':>12} {'Time (s)':>10}")
    print("-" * 55)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        # Calculate average losses and RMSE
        avg_train_loss = total_train_loss / len(train_loader)
        val_loss, rmse = evaluate_model(model, val_loader, criterion, y_scaler)
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        rmse_values.append(rmse)
        
        # Print progress
        epoch_time = time() - start_time
        print(f"{epoch+1:5d} {avg_train_loss:12.6f} {val_loss:12.6f} {rmse:12.6f} {epoch_time:10.2f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_rmse = evaluate_model(model, test_loader, criterion, y_scaler)
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Test loss: {test_loss:.6f}")
    print(f"Test RMSE: {test_rmse:.6f}")
    
    # Plot learning curves
    plot_learning_curves(train_losses, val_losses, rmse_values)
    
    return model, X_scaler, y_scaler

if __name__ == "__main__":
    csv_file = "2024-12-28_07-35-16_control_data.csv"
    try:
        model, X_scaler, y_scaler = train_drone_model(csv_file)
        
        # Example prediction
        sample_input = np.array([[0.976020, 6.775537, 0.366301, 4.789952, 
                                 0.000000, -0.000000, 0.000000, 
                                 0.000000, 9.810000, 0.000000]])
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            scaled_input = X_scaler.transform(sample_input)
            prediction = model(torch.FloatTensor(scaled_input))
            prediction = y_scaler.inverse_transform(prediction.numpy())
            print("\nPredicted omega_next values:", prediction[0])
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("\nData types in DataFrame:")
        df = pd.read_csv(csv_file)
        print(df.dtypes)