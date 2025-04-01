import torch
import torch.nn as nn
import torch.optim as optim
import random

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
        # Initialize the memory buffer and its capacity
        self.memory = []
        self.capacity = capacity

    def push(self, transition):
        # Remove the oldest memory if the buffer is full
        if len(self.memory) > self.capacity:
            self.memory.pop(0)
        # Add the new transition to the memory buffer
        self.memory.append(transition)

    def sample(self, batch_size):
        # Randomly sample a batch of transitions from the memory buffer
        return random.sample(self.memory, batch_size)

class DQNAgent:
    def __init__(self, input_dim, output_dim, lr=0.01, gamma=0.95, epsilon=1.0, epsilon_decay=0.995):
        # Initialize the DQN model
        self.model = DQN(input_dim, output_dim)
        # Define the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # Define the loss function
        self.criterion = nn.MSELoss()
        # Set the discount factor
        self.gamma = gamma
        # Set the initial exploration rate
        self.epsilon = epsilon
        # Set the exploration decay rate
        self.epsilon_decay = epsilon_decay
        # Initialize the replay memory
        self.memory = ReplayMemory(10000)

    def select_action(self, state):
        # Select a random action with probability epsilon (exploration)
        if random.random() < self.epsilon:
            return random.randint(0, 3)  # Explore (random action)
        else:
            # Select the best action based on the Q-values (exploitation)
            with torch.no_grad():
                return torch.argmax(self.model(torch.FloatTensor(state))).item()  # Exploit (best action)

    def update(self, batch_size):
        # Return if there are not enough samples in the memory
        if len(self.memory.memory) < batch_size:
            return
        
        # Sample a batch of transitions from the memory
        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert the batch data to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Compute the Q-values for the current states
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # Compute the Q-values for the next states
        next_q_values = self.model(next_states).max(1)[0].detach()
        # Compute the expected Q-values
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute the loss
        loss = self.criterion(q_values, expected_q_values)
        # Zero the gradients
        self.optimizer.zero_grad()
        # Perform backpropagation
        loss.backward()
        # Update the model parameters
        self.optimizer.step()

        # Decay the exploration rate
        self.epsilon *= self.epsilon_decay
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()  # Set the model to evaluation mode
