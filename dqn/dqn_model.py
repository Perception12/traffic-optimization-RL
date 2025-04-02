import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
from collections import deque


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to numpy for normalization
        rewards = np.array(rewards, dtype=np.float32).squeeze()

        # Normalize rewards **based only on the batch** (to prevent data leakage)
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards) + 1e-8  # Avoid division by zero
        normalized_rewards = (rewards - mean_reward) / std_reward

        # Convert to PyTorch tensors
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            # Normalized here
            torch.tensor(normalized_rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32).squeeze(),
        )

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    def __init__(self, input_dim, output_dim, lr=0.0001, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, tau=0.01):
        self.model = DQN(input_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.target_model = DQN(input_dim, output_dim)  # Add target network
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.memory = ReplayMemory(10000)
        self.tau = tau  # Soft update rate

        self.target_model.load_state_dict(
            self.model.state_dict())  # Sync weights

    def soft_update_target_network(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def select_action(self, state):
        # Select a random action with probability epsilon (exploration)
        if random.random() < self.epsilon:
            # Random action index
            action = random.randint(0, self.output_dim-1)
        else:
            # Select the best action based on the Q-values (exploitation)
            with torch.no_grad():
                action = torch.argmax(
                    self.model(torch.FloatTensor(state))).item()

        return action

    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(
            batch_size)

        # Compute Q-values
        q_values = self.model(states).gather(
            1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        expected_q_values = rewards + \
            (self.gamma * next_q_values * (1 - dones))

        loss = self.criterion(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update_target_network()

    def save_model(self, path, include_optimizer=False, metadata=None):
        """
        Enhanced model saving with additional options

        Args:
            path (str): File path to save the model
            include_optimizer (bool): Whether to save optimizer state
            metadata (dict): Additional metadata to store with the model
        """
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim
        }

        if include_optimizer:
            save_dict['optimizer_state_dict'] = self.optimizer.state_dict()

        if metadata:
            save_dict['metadata'] = metadata

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save(save_dict, path)
        print(f"Model saved successfully to {path}")

    def load_model(self, path, load_optimizer=False):
        """
        Enhanced model loading with safety checks

        Args:
            path (str): File path to load the model from
            load_optimizer (bool): Whether to load optimizer state
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")

        try:
            checkpoint = torch.load(path)

            # Verify model architecture matches
            if (checkpoint['input_dim'] != self.input_dim or
                    checkpoint['output_dim'] != self.output_dim):
                raise ValueError("Model architecture mismatch!")

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            if load_optimizer and 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(
                    checkpoint['optimizer_state_dict'])

            print(f"Model loaded successfully from {path}")
            return checkpoint.get('metadata', None)

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
