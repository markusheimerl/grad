import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Load the data
data = pd.read_csv('20250208_163908_data.csv')
X = data.iloc[:, :15].values  # First 15 columns are inputs (x0-x14)
y = data.iloc[:, 15:].values  # Last 4 columns are outputs (y0-y3)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X)
y_train = torch.FloatTensor(y)

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(15, 128)
        self.layer2 = nn.Linear(128, 4)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.layer2(x)
        return x

# Initialize the model, loss function, and optimizer
model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training parameters
num_epochs = 4000
batch_size = 32

# Training loop
for epoch in range(num_epochs):
    # Mini-batch training
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Print progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        with torch.no_grad():
            train_loss = criterion(model(X_train), y_train)
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    # Make predictions
    y_pred = model(X_train).numpy()
    y_train_np = y_train.numpy()
    
    # Calculate R² score for each output
    from sklearn.metrics import r2_score
    for i in range(4):
        r2 = r2_score(y_train_np[:, i], y_pred[:, i])
        print(f'R² score for output y{i}: {r2:.4f}')
    
    # Show sample predictions
    print("\nSample Predictions (first 15 samples):")
    print("Output\t\tPredicted\tActual\t\tDifference")
    print("-" * 60)
    
    for i in range(4):  # For each output
        print(f"\ny{i}:")
        for j in range(15):  # First 15 samples
            pred = y_pred[j, i]
            actual = y_train_np[j, i]
            diff = pred - actual
            print(f"Sample {j}:\t{pred:8.3f}\t{actual:8.3f}\t{diff:8.3f}")
        
        # Calculate and print mean absolute error for this output
        mae = np.mean(np.abs(y_pred[:, i] - y_train_np[:, i]))
        print(f"Mean Absolute Error for y{i}: {mae:.3f}")

# Save the model
# torch.save(model.state_dict(), 'model.pth')