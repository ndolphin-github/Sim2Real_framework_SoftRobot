import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load the training data
data = pd.read_csv('PyBulletData\PressureThetaMappingData.csv')

# Strip any leading/trailing whitespace characters from column names
data.columns = data.columns.str.strip()

# Print the column names to check for any issues
print("Columns in the training data:", data.columns)

# Select the first three columns as input and the next three as output
X = data[['P1', 'P2', 'P3']].values
y = data[['thetaX', 'thetaY', 'd']].values

# Print the range of the input and output data before scaling
print("Training input range before scaling:")
print("Min:", np.min(X, axis=0))
print("Max:", np.max(X, axis=0))

print("Training output range before scaling:")
print("Min:", np.min(y, axis=0))
print("Max:", np.max(y, axis=0))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)

# Print the range of the standardized training data
print("Standardized training input range:")
print("Min:", np.min(X_train_scaled, axis=0))
print("Max:", np.max(X_train_scaled, axis=0))

print("Standardized training output range:")
print("Min:", np.min(y_train_scaled, axis=0))
print("Max:", np.max(y_train_scaled, axis=0))

# Save the scalers
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')

# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create and train the model (for illustration purposes, not full training)
model = NeuralNetwork()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

# Train the model (simplified for demonstration)
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 2 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), 'pressure_theta_mapping_model.pth')

