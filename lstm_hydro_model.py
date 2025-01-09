import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
BATCH_SIZE = 64
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
        data = data = pd.read_csv(f'data/lstm_input{basin_id}.csv')
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
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()
        self.n_samples = x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

###--STEP 1: TRAINING THE MODEL--###
#### Train the model for training period
set_seed(1000000)

# Dates and basins for training
basin_list = ['01095375', '01096000', '01097000', '01103280', '01104500', '01105000']
# basin_list = pd.read_csv("MA_basins_list.csv", dtype={'basin_id':str})['basin_id'].values
start_date = date(2000, 1, 1)
end_date = date(2013, 12, 31)
n_days_train = (end_date - start_date).days + 1
n_total_days = len(basin_list) * n_days_train


#Get standard scaler from training data
features = np.zeros((n_total_days, NUM_INPUT_FEATURES), dtype=np.float32)
n = 0
for basin_id in basin_list:
    data = data = pd.read_csv(f'data/lstm_input{basin_id}.csv')
    data = data.drop(columns=['date'])
    features[n:n+n_days_train, :] = data.iloc[:n_days_train, :29].values
    n += n_days_train
scaler = StandardScaler()
scaler.fit(features)

# Load and preprocess data
features, targets = load_data(basin_list, start_date, end_date, scaler)
# x_seq, y_seq = create_sequences(features, targets, SEQUENCE_LENGTH)
#create sequences for each basin and concatenate, this is done so sequence from different basins are not mixed
x_seq, y_seq = [], []
for i in range(len(basin_list)):
    x, y = create_sequences(features[i*n_days_train:(i+1)*n_days_train], targets[i*n_days_train:(i+1)*n_days_train], SEQUENCE_LENGTH)
    x_seq.append(x)
    y_seq.append(y)
x_seq = np.concatenate(x_seq)
y_seq = np.concatenate(y_seq)

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
print(f'Total time taken to training: {time.time() - start_time:.2f} seconds')


##--STEP 2: TESTING THE MODEL--###
### 2.1: extract basin by basin model predictions

start_date = date(2000,1,1)
end_date = date(2020,12,31)

for basin_id in basin_list:
    #read data for a basin
    data = pd.read_csv(f'data/lstm_input{basin_id}.csv').drop(columns=['date'])
    #last column is the target otherwise features
    features = data.iloc[:, :29].values
    targets = data.iloc[:, [-1]].values
    #standardize features with training data scaler
    features = scaler.transform(features)

    #create sequences
    x_seq, y_seq = create_sequences(features, targets, SEQUENCE_LENGTH)
    prediction_dataset = SeqDataset(x_seq, y_seq)
    prediction_loader = DataLoader(dataset=prediction_dataset, batch_size=BATCH_SIZE, shuffle=False)

    #load model
    model = LSTMModel(NUM_INPUT_FEATURES, NUM_HIDDEN_NEURONS, NUM_HIDDEN_LAYERS, NUM_OUTPUT_FEATURES, DROPOUT_RATE).to(device)
    model.load_state_dict(torch.load('lstm_model.pth'))
    model.eval()
    #make prediction and save to csv
    all_outputs, all_targets = [], []
    with torch.no_grad():
        for inputs, target in prediction_loader:
            inputs, target = inputs.to(device), target.to(device)
            output = model(inputs)
            all_outputs.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    all_outputs = np.concatenate(all_outputs).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    #save to csv
    date_range = pd.date_range(start=start_date, end=end_date).strftime('%Y-%m-%d')
    prediction = pd.DataFrame({'date': date_range[SEQUENCE_LENGTH-1:], 'observed': all_targets, 'predicted': all_outputs})
    prediction.to_csv(f'output/lstm_output{basin_id}.csv', index=False)

print(f'Total time taken to prediction: {time.time() - start_time:.2f} seconds')


#### 2.2: calculate NSE values for each basin for training and testing periods
train_start_date, train_end_date = '2000-01-01', '2013-12-31'
test_start_date, test_end_date = '2014-01-01', '2020-12-31'

nse_train, nse_test = [], []
# Read in LSTM NSE values for each basin
for basin_id in basin_list:
    data = pd.read_csv(f'output/lstm_output{basin_id}.csv')
    data['date'] = pd.to_datetime(data['date'])
    #extract for training period
    train_data = data[(data['date'] >= train_start_date) & (data['date'] <= train_end_date)]
    #calculate NSE
    nse = 1 - sum((train_data['observed'] - train_data['predicted'])**2) / sum((train_data['observed'] - np.mean(train_data['observed']))**2)
    nse_train.append(nse)
    #extract for testing period
    test_data = data[(data['date'] >= test_start_date) & (data['date'] <= test_end_date)]
    #calculate NSE
    nse = 1 - sum((test_data['observed'] - test_data['predicted'])**2) / sum((test_data['observed'] - np.mean(test_data['observed']))**2)
    nse_test.append(nse)

# Plot NSE values
# plt.figure(figsize=(6, 4))
plt.plot(basin_list, nse_train, 'o-', label='Training NSE')
plt.plot(basin_list, nse_test, 'o-', label='Testing NSE')
plt.xlabel('Basin ID')
plt.ylabel('NSE')
# plt.ylim(0, 1)
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5)
plt.xticks(rotation=90)
plt.title('Regional LSTM Model NSE Values')
plt.tight_layout()
plt.show()
#save the plot
plt.savefig('nse_plot.png', dpi = 300)
