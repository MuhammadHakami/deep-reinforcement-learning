import torch
import torch.nn as nn
import torch.nn.functional as F

def initilize_weights(model, gain=1.0):
    nn.init.xavier_uniform_(model.weight, gain=gain)
    model.bias.data.fill_(0.009)

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        
        self.fc3 = nn.Linear(fc2_units, action_size)
        
        # initialize weights according to the Glorot and Bengio method using normal distribution and a gain.
        relu_gain=nn.init.calculate_gain('relu')
        self.fc1.apply(lambda x: initilize_weights(x, gain=relu_gain))
        self.fc2.apply(lambda x: initilize_weights(x, gain=relu_gain))
        self.fc3.apply(initilize_weights)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))

        return self.fc3(x)