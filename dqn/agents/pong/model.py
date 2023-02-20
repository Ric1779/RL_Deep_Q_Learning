import torch
from torch import Tensor
import torch.nn as nn
import numpy as np

from agents.pong.config import CartPoleConfig


class DQN(nn.Module):
    """The Deep Q-Learning Network.
    """

    def __init__(self, cfg: CartPoleConfig):
        super(DQN, self).__init__()

        self.batch_size = cfg.train.batch_size
        self.gamma = cfg.train.gamma
        self.epsilon = cfg.train.epsilon
        self.epsilon_end = cfg.train.epsilon_end
        self.anneal_length = cfg.train.anneal_length
        self.num_actions = cfg.train.num_actions
        

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, self.num_actions)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x: Tensor) -> Tensor:
        """Computes a forward pass of the network.

        Args:
            x (Tensor): The input to the network.

        Returns:
            Tensor: The output of the final layer of the network.
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def act(self, observation: Tensor) -> int:
        """Selects an action with an epsilon-greedy exploration strategy.
        0: Push cart to the left.
        1: Push cart to the right.

        Args:
            observation (Tensor): The current observation.

        Returns:
            int: The action taken by the DQN based on the observation.
        """
        if np.random.uniform(low=0.0, high=1.0) <= self.epsilon:
            # Random action.
            action = np.random.randint(low=0, high=self.num_actions)
        else:
            #print('self.action.shape{}'.format(self.num_actions))
            prediction = self(observation)
            #print('prediction shape:{}'.format(prediction))
            action = int(torch.argmax(prediction,1).item())
        return action
