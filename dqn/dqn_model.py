import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
from collections import deque

# We’ll use PyTorch to create a neural network that maps
# traffic conditions (state) to an optimal traffic light phase (action).


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        # Define the first fully connected layer
        self.fc1 = nn.Linear(input_dim, 128)
        # Define the second fully connected layer
        self.fc2 = nn.Linear(128, 64)
        # Define the third fully connected layer
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        # Apply ReLU activation function to the output of the first layer
        x = torch.relu(self.fc1(x))
        # Apply ReLU activation function to the output of the second layer
        x = torch.relu(self.fc2(x))
        # Return the raw Q-values from the third layer
        return self.fc3(x)  # No softmax (Q-values are raw)


# We train the DQN agent using experience replay and ε-greedy exploration.
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return zip(*batch)  # Unzip properly
    
    def __len__(self):
        return len(self.memory)


class DQNAgent:
    def __init__(self, input_dim, output_dim, lr=0.0001, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, tau=0.01):
        self.model = DQN(input_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.target_model = DQN(input_dim, output_dim)  # Add target network
        self.target_model.load_state_dict(self.model.state_dict())  # Sync weights
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.memory = ReplayMemory(10000)
        self.tau = tau  # Soft update rate

    def soft_update_target_network(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def select_action(self, state):
        # Select a random action with probability epsilon (exploration)
        if random.random() < self.epsilon:
            action = random.randint(0, 3)  # Random action index
        else:
            # Select the best action based on the Q-values (exploitation)
            with torch.no_grad():
                action = torch.argmax(
                    self.model(torch.FloatTensor(state))).item()

        return action

    def update(self, batch_size):
        if len(self.memory.memory) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # Ensure correct numpy array types
        actions = np.array(actions, dtype=np.int64)  # Convert to int64
        # Convert and remove extra dimensions
        rewards = np.array(rewards, dtype=np.float32).squeeze()
        # Convert and remove extra dimensions
        dones = np.array(dones, dtype=np.float32).squeeze()
        
        mean_reward = np.mean([r for (_, _, r, _, _) in self.memory.memory])
        std_reward = np.std([r for (_, _, r, _, _) in self.memory.memory]) + 1e-8
        normalized_rewards = (rewards - mean_reward) / std_reward

        # Convert to PyTorch tensors
        states = torch.as_tensor(np.array(states), dtype=torch.float32)
        next_states = torch.as_tensor(
            np.array(next_states), dtype=torch.float32)

        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(normalized_rewards, dtype=torch.float32)
        # Convert to float tensor and remove extra dimensions
        dones = torch.tensor(dones)
        
        

        # Compute Q-values
        q_values = self.model(states).gather(
            1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        expected_q_values = rewards + \
            (self.gamma * next_q_values * (1 - dones)).detach()
        
        self.soft_update_target_network()

        loss = self.criterion(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(0.05, self.epsilon * self.epsilon_decay)


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
            checkpoint = torch.load(path, weights_only=True)

            # Verify model architecture matches
            if (checkpoint['input_dim'] != self.input_dim or
                    checkpoint['output_dim'] != self.output_dim):
                raise ValueError("Model architecture mismatch!")

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            if load_optimizer and 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            print(f"Model loaded successfully from {path}")
            return checkpoint.get('metadata', None)

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
