import torch
from lib import Variable

class DQL():
    ''' Deep Q Neural Network class. '''
    def __init__(self, state_dim, action_dim, hidden_layer_dim=64, learning_rate=0.05):
            self.criterion = torch.nn.MSELoss()
            self.model = torch.nn.Sequential(
                            torch.nn.Linear(state_dim, hidden_layer_dim),
                            torch.nn.LeakyReLU(),
                            torch.nn.Linear(hidden_layer_dim, hidden_layer_dim),
                            torch.nn.LeakyReLU(),
                            torch.nn.Linear(hidden_layer_dim, action_dim)
                    )
            self.optimizer = torch.optim.Adam(self.model.parameters(), learning_rate)

    def update(self, state, y):
            """Update the weights of the network given a training sample. """
            y_pred = self.model(torch.Tensor(state))
            loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, state):
            """ Compute Q values for all actions using the DQL. """
            with torch.no_grad():
                return self.model(torch.Tensor(state))