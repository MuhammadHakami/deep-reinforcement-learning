import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

def initilize_weights(model, gain=1.0):
    nn.init.xavier_uniform_(model.weight, gain=gain)
    model.bias.data.fill_(0.009)

class ActorCritic(nn.Module):

    def __init__(self, state_size, action_size, seed=42, fc1_units=160):
        super(ActorCritic, self).__init__()

        self.seed = torch.manual_seed(seed)

        # Actor model
        self.actor= nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.Mish(),
            nn.Linear(fc1_units, round((1-0.2)*fc1_units)),
            nn.Mish(),
            nn.Linear(round((1-0.2)*fc1_units), action_size),
            nn.Tanh()
            )


        self.fcs1 = nn.Linear(state_size, fc1_units)
        self.mish1= nn.Mish()
        self.fc2 = nn.Linear(fc1_units+action_size, fc1_units)
        self.mish2= nn.Mish()
        self.fc3 = nn.Linear(fc1_units, round((1-0.2)*fc1_units))
        self.mish3= nn.Mish()
        self.fc4 = nn.Linear(round((1-0.2)*fc1_units), 1)

        # self.reset_parameters()
        # self.critic= nn.Sequential(
        #     nn.Linear(state_size+action_size, fc1_units),
        #     nn.LeakyReLU(),
        #     nn.Linear(fc1_units, round(fc1_units/2)),
        #     nn.LeakyReLU(),
        #     nn.Linear(round(fc1_units/2), round(fc1_units/4)),
        #     nn.LeakyReLU(),
        #     nn.Linear(round(fc1_units/4), 1),
        #     )
        self.reset_parameters()
        self.std = nn.Parameter(torch.zeros(action_size))

    def forward(self, x):
        actions = self.actor(x)
        dist = torch.distributions.Normal(actions, F.softplus(self.std))
        action = torch.clamp(dist.sample(), -1, 1)

        xs =self.mish1(self.fcs1(x))
        x = torch.cat((xs, action), dim=1)
        x =self.mish2(self.fc2(x))
        x =self.mish3(self.fc3(x))
        return action, dist, self.fc4(x)

    def reset_parameters(self):
        # initialize weights according to the Glorot and Bengio method using normal distribution and a gain.
        leaky_relu_gain=nn.init.calculate_gain('leaky_relu')
        tanh_gain=nn.init.calculate_gain('tanh')
        # mish_gain=nn.init.calculate_gain('mish')

        self.actor[0].apply(lambda x: initilize_weights(x, gain=leaky_relu_gain))
        self.actor[2].apply(lambda x: initilize_weights(x, gain=leaky_relu_gain))
        self.actor[4].apply(lambda x: initilize_weights(x, gain=tanh_gain))


        self.fcs1.apply(lambda x: initilize_weights(x, gain=leaky_relu_gain))
        self.fc2.apply(lambda x: initilize_weights(x, gain=leaky_relu_gain))
        self.fc3.apply(lambda x: initilize_weights(x, gain=leaky_relu_gain))
        self.fc4.apply(initilize_weights)





def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_units=160):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc_units)
        self.mish1=nn.Mish()
        self.fc2 = nn.Linear(fc_units, round((1-0.2)*fc_units))
        self.mish2=nn.Mish()
        self.fc3 = nn.Linear(round((1-0.2)*fc_units), action_size)
        self.reset_parameters()
        

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.mish1(self.fc1(state))
        x = self.mish2(self.fc2(x))
        return torch.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=160):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.mish1= nn.Mish()
        self.fc2 = nn.Linear(fcs1_units+action_size, fcs1_units)
        self.mish2= nn.Mish()
        self.fc3 = nn.Linear(fcs1_units, round((1-0.2)*fcs1_units))
        self.mish3= nn.Mish()
        self.fc4 = nn.Linear(round((1-0.2)*fcs1_units), 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs =self.mish1(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x =self.mish2(self.fc2(x))
        x =self.mish3(self.fc3(x))
        return self.fc4(x)
