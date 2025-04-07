import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)  # Action logits
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)  # Action probabilities

class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Value estimates for each edge
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)  # Per-edge value estimates

class A2CAgent:
    def __init__(self, input_dim, output_dim, actor_lr=0.0002, critic_lr=0.0005, gamma=0.95):
        self.actor = Actor(input_dim, output_dim)
        self.critic = Critic(input_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.critic_loss_fn = nn.SmoothL1Loss()
        self.min_reward = np.inf
        self.max_reward = -np.inf

    def select_action(self, state):
        state = torch.FloatTensor(state)
        probs = self.actor(state)
        action = torch.multinomial(probs, num_samples=1).item()
        return action, probs.squeeze(0)[action]  # Ensure correct indexing

    def update(self, state, action, reward, next_state, done):
        
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)

        value = self.critic(state)  # Scalar value estimate
        next_value = self.critic(next_state).detach()

        # Corrected advantage calculation
        target = reward + self.gamma * next_value * (1 - done)
        advantage = target - value

        # Corrected log probability selection
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        log_prob = dist.log_prob(torch.tensor(action))
        entropy = dist.entropy()

        # Actor loss: maximize advantage (policy gradient)
        actor_loss = -log_prob * advantage.detach() - 0.01 * entropy

        # Critic loss: SmoothL1 Loss
        critic_loss = self.critic_loss_fn(value, target.detach())

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def save_model(self, actor_path, critic_path):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(actor_path), exist_ok=True)
        os.makedirs(os.path.dirname(critic_path), exist_ok=True)
        
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        print(f"Models saved successfully to {actor_path} and {critic_path}")

    def load_model(self, actor_path, critic_path):
        if not (os.path.exists(actor_path) or os.path.exists(critic_path)):
            raise FileNotFoundError(f"No model found at {actor_path} or {critic_path}")
        
        try:
            self.actor.load_state_dict(torch.load(actor_path))
            self.critic.load_state_dict(torch.load(critic_path))
            self.actor.eval()
            self.critic.eval()
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise