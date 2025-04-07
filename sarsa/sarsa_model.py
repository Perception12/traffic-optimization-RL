import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import numpy as np

class SARSA(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SARSA, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Q-values for all actions

class SARSAAgent:
    def __init__(self, input_dim, output_dim, lr=0.0001, gamma=0.95, epsilon=1.0, epsilon_decay=0.995):
        self.model = SARSA(input_dim, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_reward = np.inf
        self.max_reward = -np.inf

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.output_dim-1)  # Explore
        else:
            with torch.no_grad():
                return torch.argmax(self.model(torch.FloatTensor(state))).item()  # Exploit

    def update(self, state, action, reward, next_state, next_action, done):
        self.min_reward = min(self.min_reward, reward)
        self.max_reward = max(self.max_reward, reward)
        normalized_reward = (reward - self.min_reward) / (self.max_reward - self.min_reward + 1e-5)

        state_tensor = torch.tensor(state, dtype=torch.float32)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        
        q_values = self.model(state_tensor)
        next_q_values = self.model(next_state_tensor)

        q_value = q_values[action]
        next_q_value = next_q_values[next_action] if not done else 0

        target = normalized_reward + self.gamma * next_q_value
        loss = self.criterion(q_value, torch.tensor(target, dtype=torch.float32))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, path):
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save(self.model.state_dict(), path)
        print(f"Model saved successfully to {path}")

    def load_model(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")

        try:
            checkpoint = torch.load(path)

            self.model.load_state_dict(checkpoint)
            self.model.eval()

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
