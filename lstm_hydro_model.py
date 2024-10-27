import numpy as np
import pandas as pd
from datetime import date
import time
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

start_time = time.time()
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
NUM_INPUT_FEATURES = 29
NUM_OUTPUT_FEATURES = 1
NUM_EPOCHS = 30
NUM_HIDDEN_LAYERS = 1
SEQUENCE_LENGTH = 365
NUM_HIDDEN_NEURONS = 256
BATCH_SIZE = 24
LEARNING_RATE = 0.0005
DROPOUT_RATE = 0.4

# Function to set random seed
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# Function to load and preprocess data
def load_data(basin_list, start_date, end_date, scaler):
    n_days_train = (end_date - start_date).days + 1
    n_total_days = len(basin_list) * n_days_train

    features = np.zeros((n_total_days, NUM_INPUT_FEATURES), dtype=np.float32)
    targets = np.zeros((n_total_days, NUM_OUTPUT_FEATURES), dtype=np.float32)

    n = 0
    for basin_id in basin_list:
        data = pd.read_csv(f'data/lstm_input{basin_id}.csv')
        data = data.drop(columns=['date'])
        # data_size = len(data)
        features[n:n+n_days_train, :] = data.iloc[:n_days_train, :29].values # 29 features
        targets[n:n+n_days_train, :] = data.iloc[:n_days_train, [-1]].values # last column is the target
        n += n_days_train
    #standardize features with training data scaler
    features = scaler.transform(features)

    return features, targets

# Function to create LSTM input sequences
def create_sequences(features, targets, sequence_length):
    n_days = len(features) - sequence_length + 1
    x_sequences = []
    y_sequences = []

    for i in range(n_days):
        x_seq = features[i:i+sequence_length]
        y_seq = targets[i+sequence_length-1]
        x_sequences.append(x_seq)
        y_sequences.append(y_seq)

    return np.array(x_sequences), np.array(y_sequences)

# LSTM Model Definition
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_neurons, num_layers, output_size, dropout_rate):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_neurons, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_neurons, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(NUM_HIDDEN_LAYERS, x.size(0), NUM_HIDDEN_NEURONS).to(device)
        c0 = torch.zeros(NUM_HIDDEN_LAYERS, x.size(0), NUM_HIDDEN_NEURONS).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return self.relu(out)

# Dataset Class
class SeqDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.n_samples = x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

#### Train the model for training period
set_seed(1000000)

# Dates and basins for training
start_date = date(2000, 1, 1)
end_date = date(2013, 12, 31)
basin_list = ['01095375', '01096000', '01097000', '01103280', '01104500', '01105000']

#Get standard scaler from training data
n_days_train = (end_date - start_date).days + 1
n_total_days = len(basin_list) * n_days_train
features = np.zeros((n_total_days, NUM_INPUT_FEATURES), dtype=np.float32)
n = 0
for basin_id in basin_list:
    data = pd.read_csv(f'data/lstm_input{basin_id}.csv')
    data = data.drop(columns=['date'])
    features[n:n+n_days_train, :] = data.iloc[:n_days_train, :29].values
    n += n_days_train
scaler = StandardScaler()
scaler.fit(features)

# Load and preprocess data
features, targets = load_data(basin_list, start_date, end_date, scaler)
x_seq, y_seq = create_sequences(features, targets, SEQUENCE_LENGTH)

# Dataset and DataLoader
train_dataset = SeqDataset(x_seq, y_seq)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model
model = LSTMModel(NUM_INPUT_FEATURES, NUM_HIDDEN_NEURONS, NUM_HIDDEN_LAYERS, NUM_OUTPUT_FEATURES, DROPOUT_RATE).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
for epoch in range(NUM_EPOCHS):
    model.train()
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        end_time = time.time()
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Time: {end_time - start_time:.2f} seconds')

# Save the model
torch.save(model.state_dict(), 'lstm_model.pth')


#### Test the model for testing period

start_date = date(2014, 1, 1)
end_date = date(2020, 12, 31)

features, targets = load_data(basin_list, start_date, end_date, scaler)
x_seq, y_seq = create_sequences(features, targets, SEQUENCE_LENGTH)

test_dataset = SeqDataset(x_seq, y_seq)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = LSTMModel(NUM_INPUT_FEATURES, NUM_HIDDEN_NEURONS, NUM_HIDDEN_LAYERS, NUM_OUTPUT_FEATURES, DROPOUT_RATE).to(device)
model.load_state_dict(torch.load('lstm_model.pth'))
model.eval()

all_outputs, all_targets = [], []
with torch.no_grad():
    for inputs, target in test_loader:
        inputs, target = inputs.to(device), target.to(device)
        output = model(inputs)
        all_outputs.append(output.cpu().numpy())
        all_targets.append(target.cpu().numpy())
all_outputs = np.concatenate(all_outputs).flatten()
all_targets = np.concatenate(all_targets).flatten()

#find test nse values for each basin
n_days_test = (end_date - start_date).days + 1
test_basin_size = n_days_test
for i in range(len(basin_list)):
    test_basin_outputs = all_outputs[i*test_basin_size:(i+1)*test_basin_size]
    test_basin_targets = all_targets[i*test_basin_size:(i+1)*test_basin_size]
    nse = 1 - np.sum((test_basin_outputs-test_basin_targets)**2)/np.sum((test_basin_targets-np.mean(test_basin_targets))**2)
    print(f'Basin {basin_list[i]} NSE: {nse}')
