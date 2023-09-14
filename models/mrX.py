import torch
import torch.nn as nn
import torch.optim as optim

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.learning_rate = 1e-3
        self.columns = 1415

        self.hidden1 = int(self.columns / 2)
        # self.hidden2 = int(self.columns / 2)
        # self.hidden3 = int(self.columns / 4)
        # self.hidden4 = int(self.columns / 4)
        self.out = 1

        self.fc1 = nn.Linear(self.columns, self.hidden1)
        self.fc_out = nn.Linear(self.hidden1, self.out)

        self.relu = nn.ReLU()
        
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc_out(x)
        return x

    def optimize(self, x, y):
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        predictions = self(x)
        loss = self.loss_function(predictions, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, episode):
        # Saving the model state dictionary to a file
        torch.save(self.state_dict(), f'models/x_models/model_{episode}.pth')

    def restore(self, episode):
        # Loading a model state dictionary from a file
        self.load_state_dict(torch.load(f'models/x_models/model_{episode}.pth'))
