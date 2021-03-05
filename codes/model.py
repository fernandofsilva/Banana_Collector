import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model.

    This class construct the model.
    """

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128, fc3_units=32):
        """ Initialize parameters and build model.

        Args:
            state_size: Integer. Dimension of each state
            action_size: Integer. Dimension of each action
            seed: Integer. Value to set the seed of the model
            fc1_units: Integer. Number of nodes in first fully connect hidden layer
            fc2_units: Integer. Number of nodes in second fully connect hidden layer
            fc3_units: Integer. Number of nodes in third fully connect hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)

    def __repr__(self):
        return 'QNetwork'

    def __str__(self):
        return 'QNetwork'

    def forward(self, state):
        """Defines the computation performed at every call.

        Args:
            state: A tensor with the state values

        Returns:
            A tensor if there is a single output, or a list of tensors if there
                are more than one outputs.
        """
        # Define the hidden layers
        hidden = F.relu(self.fc1(state))
        hidden = F.relu(self.fc2(hidden))
        hidden = F.relu(self.fc3(hidden))

        return self.fc4(hidden)
