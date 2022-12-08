import torch
from torch.autograd import Variable
import numpy as np


# Deep Q Neural Network class
class DQN(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, learning_rate):
        super(DQN, self).__init__()
        self.state_dim = state_dim
        # Policy Network
        self.model = torch.nn.Sequential(
                torch.nn.Linear(state_dim, hidden_dim),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim*2),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(hidden_dim*2, action_dim)
        )
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), learning_rate)

    # Update the weights of the network given a training sample
    def update(self, state, y):
        y_pred = self.model(torch.tensor(np.array(state[0].tolist()), dtype=torch.float32))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # Compute Q values for all actions using the model
    def predict(self, state):
        with torch.no_grad():
                return self.model(torch.tensor(np.array(state[0].tolist()), dtype=torch.float32))  # .max(1)[0].detach()
