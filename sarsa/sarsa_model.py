import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

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
    def __init__(self, input_dim, output_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995):
        self.model = SARSA(input_dim, output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)  # Explore
        else:
            with torch.no_grad():
                return torch.argmax(self.model(torch.FloatTensor(state))).item()  # Exploit

    def update(self, state, action, reward, next_state, next_action, done):
        state_tensor = torch.FloatTensor(state)
        next_state_tensor = torch.FloatTensor(next_state)
        
        q_values = self.model(state_tensor)
        next_q_values = self.model(next_state_tensor)

        q_value = q_values[action]
        next_q_value = next_q_values[next_action] if not done else 0

        target = reward + self.gamma * next_q_value
        loss = self.criterion(q_value, torch.tensor(target))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon *= self.epsilon_decay

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
