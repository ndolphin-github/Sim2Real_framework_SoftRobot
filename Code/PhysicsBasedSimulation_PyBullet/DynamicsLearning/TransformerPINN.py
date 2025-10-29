import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.optim import Adam
import joblib

# 1. Load the dataset
data_path = 'Pybullet_dataset/Pybullet_joint_to_pos.csv'
data = pd.read_csv(data_path)

# Downcast data to float32 to save memory
data = data.astype(np.float32)

# 2. Normalize the data
scaler = MinMaxScaler()
data[data.columns] = scaler.fit_transform(data[data.columns])

# Save the scaler for later use
joblib.dump(scaler, 'scaler.pkl')
# 3. Split data into training and validation sets
train_data, val_data = train_test_split(data, test_size=0.2, shuffle=False, random_state=42)

# 4. Define a custom Dataset class that processes data on-the-fly
class SequenceDataset(Dataset):
    def __init__(self, data, sequence_length, input_columns, output_columns):
        self.data = data
        self.sequence_length = sequence_length
        self.input_columns = input_columns
        self.output_columns = output_columns

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        input_seq = self.data[self.input_columns].values[idx:idx + self.sequence_length]
        output_seq = self.data[self.output_columns].values[idx + self.sequence_length]
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(output_seq, dtype=torch.float32)

# Define columns for input and output
input_columns = ['thetaX', 'thetaY', 'd', 'Fx', 'Fy', 'Fz']
output_columns = ['P1', 'P2', 'P3']

# Create datasets and dataloaders
sequence_length = 30
train_dataset = SequenceDataset(train_data, sequence_length, input_columns, output_columns)
val_dataset = SequenceDataset(val_data, sequence_length, input_columns, output_columns)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 5. Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, nhead=4, num_layers=2, hidden_size=64, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.transformer(x)
        out = self.fc2(x[:, -1, :])  # Take the output of the last time step
        return out

# Initialize the Transformer model
input_size = len(input_columns)
output_size = len(output_columns)
model = TransformerModel(input_size=input_size, output_size=output_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 6. Define PINN Loss and Total Loss
class PINNLoss(nn.Module):
    def __init__(self):
        super(PINNLoss, self).__init__()

    def forward(self, predicted_output, ground_truth_output, physics_residual):
        data_loss = nn.MSELoss()(predicted_output, ground_truth_output)
        physics_loss = torch.mean(physics_residual ** 2)
        return data_loss + 0.1 * physics_loss  # Weight for physics loss

# Initialize the loss and optimizer
criterion = nn.MSELoss()  # Data-driven loss
pinn_loss_fn = PINNLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# 7. Training loop
epochs = 30

def train_model(model, optimizer, train_loader, val_loader, device, criterion, pinn_loss_fn):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_inputs, batch_outputs in train_loader:
            batch_inputs, batch_outputs = batch_inputs.to(device), batch_outputs.to(device)
            optimizer.zero_grad()

            # Forward pass
            predictions = model(batch_inputs)

            # Data loss
            data_loss = criterion(predictions, batch_outputs)

            # Example Physics-Informed Loss (Dummy for demonstration)
            physics_residual = torch.zeros_like(batch_outputs)  # Replace with actual physics residual calculation
            total_loss = pinn_loss_fn(predictions, batch_outputs, physics_residual)

            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_outputs in val_loader:
                val_inputs, val_outputs = val_inputs.to(device), val_outputs.to(device)
                val_predictions = model(val_inputs)
                val_loss += criterion(val_predictions, val_outputs).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")

    # Save the trained model
    torch.save(model.state_dict(), 'transformer_pinn_model.pth')

# Train the model
train_model(model, optimizer, train_loader, val_loader, device, criterion, pinn_loss_fn)
