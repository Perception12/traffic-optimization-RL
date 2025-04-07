from dqn_model import DQNAgent
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from traffic_environment import TrafficEnv
from config import config

scenario = 1
# Defining the simulation paths
config_path = os.path.abspath(f"../scenarios/test_scenario/four_way_simulation.sumocfg")
output_path = config.test_output_paths[scenario-1]


# Initialize traffic environment
env = TrafficEnv(
    config_path=config_path,
    output_path=output_path,
    scenario_name=config.scenario_names[scenario-1],
    max_steps=config.max_steps,)

# Intialize DQN Agent
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
agent = DQNAgent(input_dim, output_dim, epsilon=0.05)  # Set epsilon to a low value for testing

# Load the trained model
agent.load_model(f"models/final_dqn_model_scenario_{scenario}.pth")

# Run the simulation for 1000 steps
state = env.reset()
done = False
moving_avg_rewards = []
window_size = 50  # For moving average
reward_history = deque(maxlen=window_size)

while not done:
    action = agent.select_action(state) # Choose action from policy
    next_state, reward, done, _ = env.step(action) # Step through environment
    state = next_state

    reward_history.append(reward)
    moving_avg = np.mean(reward_history)
    moving_avg_rewards.append(moving_avg)
env.close()

# Plot test rewards
plt.figure(figsize=(12, 6))
plt.plot(moving_avg_rewards, label=f'Step Reward)')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('DQN Progress')
plt.legend()
plt.savefig(f"results/test_curve_scenario_{scenario}.png")
plt.close()

print("DQN testing completed. Results saved ")


