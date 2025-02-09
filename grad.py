import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import glob
import os

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1_weight = torch.nn.Parameter(torch.randn(1024, 16, device=device) / np.sqrt(16))
        self.fc2_weight = torch.nn.Parameter(torch.randn(4, 1024, device=device) / np.sqrt(1024)) 

    def forward(self, x):
        x = F.linear(x, self.fc1_weight, bias=None)
        x = F.relu(x)
        x = F.linear(x, self.fc2_weight, bias=None)
        return x

def load_csv(filename):
    X = []
    y = []
    
    with open(filename, 'r') as f:
        # Skip header
        next(f)
        
        for line in f:
            values = [float(x) for x in line.strip().split(',')]
            X.append(values[:16])
            y.append(values[16:])
    
    return np.array(X), np.array(y)

# Load the data
X, y = load_csv(glob.glob('*_data.csv')[0])

# Convert to PyTorch tensors and move to GPU
X_train = torch.FloatTensor(X).to(device)
y_train = torch.FloatTensor(y).to(device)

# Initialize model and move to GPU
model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Training parameters
num_epochs = 10000

# Training loop
model.train()
for epoch in range(num_epochs):
    outputs = model(X_train)
    loss = F.mse_loss(outputs, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.8f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    # Move predictions back to CPU for numpy operations
    y_pred = model(X_train).cpu().numpy()
    y_train_np = y_train.cpu().numpy()
    
    def r2_score(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    for i in range(4):
        r2 = r2_score(y_train_np[:, i], y_pred[:, i])
        print(f'R² score for output y{i}: {r2:.8f}')
    
    print("\nSample Predictions (first 15 samples):")
    print("Output\t\tPredicted\tActual\t\tDifference")
    print("-" * 60)
    
    for i in range(4):
        print(f"\ny{i}:")
        for j in range(15):
            pred = y_pred[j, i]
            actual = y_train_np[j, i]
            diff = pred - actual
            print(f"Sample {j}:\t{pred:8.3f}\t{actual:8.3f}\t{diff:8.3f}")
        
        mae = np.mean(np.abs(y_pred[:, i] - y_train_np[:, i]))
        print(f"Mean Absolute Error for y{i}: {mae:.3f}")