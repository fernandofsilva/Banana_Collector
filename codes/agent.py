import random
import numpy as np
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

# Determine if the GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """Interacts with and learns from the environment."""
    def __init__(self,
                 state_size,
                 action_size,
                 buffer_size,
                 batch_size,
                 gamma,
                 tau,
                 lr,
                 update_every,
                 seed):
        """Initialize an Agent object.

        Args:
            state_size: Integer. Dimension of each state
            action_size: Integer. Dimension of each action
            buffer_size: Integer. Replay buffer size
            batch_size: Integer. Mini-batch size
            gamma: Float. Discount factor
            tau: Float. For soft update of target parameters
            lr: Float. Learning rate
            update_every: Integer. How often to update the network
            seed: Integer. Random seed
        """
        # Environment parameters
        self.state_size = state_size
        self.action_size = action_size

        # Q-Learning
        self.gamma = gamma

        # Q-Network
        self.model_local = QNetwork(state_size, action_size, seed).to(device)
        self.model_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.model_local.parameters(), lr=lr)
        self.loss_fn = F.mse_loss
        self.tau = tau
        self.update_every = update_every
        self.batch_size = batch_size

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)

        # Initialize time step (for updating every update_every steps)
        self.t_step = 0

        # Set seed
        self.seed = random.seed(seed)

    def __str__(self):
        pass

    def __repr__(self):
        pass

    @staticmethod
    def env_step(env, action, brain_name):
        """Apply an action and return the state, reward and done.

        Args:
            env: unity environment
            action: Integer. Action to be done in the environment
            brain_name: String. Name of the agent of the unity environment

        Returns:
            A tuple of three items with
            next_state: List. Contains the next state returned,
            reward: Float. Number of the reward returned.
            done: Boolean. Indication if the episode ends.
        """

        # send the action to the environment
        env_info = env.step(action)[brain_name]

        # get the next state
        next_state = env_info.vector_observations[0]

        # Get the reward
        reward = env_info.rewards[0]

        # is it the episode ended?
        done = env_info.local_done[0]

        return next_state, reward, done

    def step(self, state, action, reward, next_state, done):
        """Save state on buffer and trigger learn according to update_every

        Args:
            state: The previous state of the environment
            action: Integer. Previous action selected by the agent
            reward: Float. Reward value
            next_state: The current state of the environment
            done: Boolean. Whether the episode is complete
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every update_every time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:

            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Args:
            state: A array like object or list with states
            eps: Float. Random value for epsilon-greedy action selection

        Returns:
            An action selected by the network or by the epsilon-greedy method
        """
        # Reshape state
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        # Set model to prediction
        self.model_local.eval()

        # Predict action
        with torch.no_grad():
            action_values = self.model_local(state)

        # Set model to training
        self.model_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values)
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.

        Args:
            experiences: Tuple. Content of tuple (s, a, r, s', done)
        """
        # Unpack experiences
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        q_targets_next = self.model_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        # Get expected Q values from local model
        q_expected = self.model_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(q_expected, q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update()

    def soft_update(self):
        """Soft update model parameters.

        The model is update using:
            θ_target = τ * θ_local + (1 - τ) * θ_target
        """

        for target_param, local_param in zip(self.model_target.parameters(), self.model_local.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Args:
            action_size: Integer. Dimension of each action
            buffer_size: Integer. Maximum size of buffer
            batch_size: Integer. Size of each training batch
            seed: Integer. Random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def __str__(self):
        pass

    def __repr__(self):
        pass

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory.

        Args:
            state: The previous state of the environment
            action: Integer. Previous action selected by the agent
            reward: Float. Reward value
            next_state: The current state of the environment
            done: Boolean. Whether the episode is complete
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])
        ).float().to(device)

        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])
        ).long().to(device)

        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])
        ).float().to(device)

        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])
        ).float().to(device)

        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
        ).float().to(device)

        return states, actions, rewards, next_states, dones
