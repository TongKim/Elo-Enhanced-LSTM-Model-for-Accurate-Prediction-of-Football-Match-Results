import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from apex.optimizers import FusedAdam
from torch.optim.lr_scheduler import StepLR

# Read the dataset
data = pd.read_csv('E02.csv')

# Convert match dates
data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = (data['Date'] - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')

# Select required features
features = ['HS3A', 'HST4A', 'AS3A', 'AST3A', 'HomeElo', 'AwayElo']

# Extract features and labels
X = data[features].values
y = data[['FTHG', 'FTAG']].values

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)
device = torch.device("cuda")

# Split train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.204558, random_state=42)
# Wrap train set data and labels into a dataset
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float16), torch.tensor(y_train, dtype=torch.float16))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float16), torch.tensor(y_test, dtype=torch.float16))
# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
torch.backends.cudnn.benchmark = True

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False, dropout=0, bias=True).cuda()
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device).half()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=device).half()
        out, _ = self.lstm(x.half(), (h0, c0))
        out = self.fc(out[:, -1, :]).half()
        return out

# Model parameters
input_size = 6
hidden_size = 512
output_size = 2
num_layers = 2

# Build the model and use DataParallel
model = LSTMModel(input_size, hidden_size, output_size, num_layers).to(device)
model.half() # Convert the model to half-precision
model = model.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss().half()
optimizer = FusedAdam(model.parameters(), lr=0.0000001, betas=(0.9, 0.999))

import numpy as np
from tqdm import tqdm
from sklearn.metrics import r2_score

# Hyperparameters
num_epochs = 100
batch_size = 1

train_losses = []
val_losses = []
accuracies = []

for epoch in range(num_epochs):
    train_loss = 0
    model.train()
    for i in tqdm(range(0, len(X_train), batch_size)):
        inputs = torch.tensor(X_train[i:i+batch_size], dtype=torch.float16).unsqueeze(1).to(device).half()
        targets = torch.tensor(y_train[i:i+batch_size], dtype=torch.float16).to(device).half()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets).half()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
    train_loss /= len(X_train)
    train_losses.append(train_loss)
    print('Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch+1, num_epochs, train_loss))

    with torch.no_grad():
        predictions = []
        val_loss = 0
        total = len(X_test)
        correct = 0
        for i in tqdm(range(0, len(X_test), batch_size)):
            inputs = torch.tensor(X_test[i:i+batch_size], dtype=torch.float16).unsqueeze(1).to(device).half()
            targets = torch.tensor(y_test[i:i+batch_size], dtype=torch.float16).to(device).half()
            outputs = model(inputs).half()
            predictions.append(outputs.cpu().numpy())
            loss = criterion(outputs, targets).half()
            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(torch.round(outputs), dim=1)
            targets = torch.argmax(targets, dim=1)
            predicted = predicted.long()
            targets = targets.long()
            correct += (predicted == targets).sum().item()
        val_loss /= len(X_test)
        accuracy = 100 * correct / total
        val_losses.append(val_loss)
        accuracies.append(accuracy)
        predictions = np.concatenate(predictions)
        r2 = r2_score(y_test, predictions)
        print('Val Loss: {:.4f}'.format(val_loss))
        print('R^2 on validation set: {:.2f}'.format(r2))
        print('Accuracy on validation set: {:.2f}%'.format(accuracy))

import matplotlib.pyplot as plt

# Visualize training loss
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='red')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Predict data
predictions = []
with torch.no_grad():
    for i in range(0, len(X_test), batch_size):
        inputs = torch.tensor(X_test[i:i+batch_size], dtype=torch.float16).unsqueeze(1).to(device).half()
        outputs = model(inputs)
        predictions.append(outputs.cpu().numpy())

# Store predicted data in DataFrame
predictions = np.concatenate(predictions)
predicted_df = pd.DataFrame(predictions, columns=['HomePredict', 'AwayPredict'])

# Merge predicted data with original data
data = data.join(predicted_df)

# Export as CSV file
data.to_csv('predicted_data.csv', index=False)

