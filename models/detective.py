import torch
import torch.nn as nn
import torch.optim as optim

class DetectiveModel(nn.Module):

    def __init__(self, device: str = 'cpu'):
        super(DetectiveModel, self).__init__()
        self.learning_rate = 1e-3
        self.columns = 126

        self.hidden1 = int(self.columns / 2)
        self.hidden2 = int(self.columns / 2)
        self.hidden3 = int(self.columns / 2)
        # self.hidden4 = int(self.columns / 2)
        # self.hidden5 = int(self.columns / 4)
        # self.hidden6 = int(self.columns / 4)
        # self.hidden7 = int(self.columns / 4)
        # self.hidden8 = int(self.columns / 4)
        # self.hidden9 = int(self.columns / 8)
        # self.hidden10 = int(self.columns / 8)
        # self.hidden11 = int(self.columns / 8)
        # self.hidden12 = int(self.columns / 8)
        # self.hidden13 = int(self.columns / 8)
        # self.hidden14 = int(self.columns / 16)
        # self.hidden15 = int(self.columns / 16)
        # self.hidden16 = int(self.columns / 16)
        # self.hidden17 = int(self.columns / 16)
        # self.hidden18 = int(self.columns / 16)
        # self.hidden19 = int(self.columns / 32)
        # self.hidden20 = int(self.columns / 32)
        self.out = 1

        self.fc1 = nn.Linear(self.columns, self.hidden1)
        self.fc2 = nn.Linear(self.hidden1, self.hidden2)
        self.fc3 = nn.Linear(self.hidden2, self.hidden3)
        self.fc_out = nn.Linear(self.hidden3, self.out)

        self.relu = nn.ReLU()
        
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.device = device
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc_out(x)
        return x

    def optimize(self, x, y):
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device)
        predictions = self(x)
        loss = self.loss_function(predictions, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, x):
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            output = self(x_tensor)
        return output.numpy()

    def save(self, episode):
        # Saving the model state dictionary to a file
        torch.save(self.state_dict(), f'models/detective_models/model_{episode}.pth')

    def restore(self, episode):
        # Loading a model state dictionary from a file
        self.load_state_dict(torch.load(f'models/detective_models/model_{episode}.pth'))
