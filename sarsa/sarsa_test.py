from sarsa_model import SARSAAgent  # Import SARSA agent
import os
import sys
from config import config
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from traffic_environment import TrafficEnv

scenario = 1
# Defining the simulation paths
config_path = os.path.abspath(f"../scenarios/test_scenario/four_way_simulation.sumocfg")
output_path = config.test_output_paths[scenario-1]

# Initialize traffic environment
env = TrafficEnv(
    config_path=config_path, 
    scenario_name=config.scenario_names[scenario-1], 
    output_path=output_path, 
    max_steps=config.max_steps)

# Initialize the A2CAgent
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
agent = SARSAAgent(input_dim, output_dim)  

agent.load_model(f"models/best_sarsa_model_scenario_{scenario}.pth")

state = env.reset()
total_reward = 0
done = False
moving_avg_rewards = []
window_size = 50  # For moving average
reward_history = deque(maxlen=window_size)
    
    
while not done:
    action = agent.select_action(state)
    next_state, reward, done, _ = env.step(action)
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
plt.title('SARSA Progress')
plt.legend()
plt.savefig(f"results/test_curve_scenario_{scenario}.png")
plt.close()

print(f"SARSA simulation completed for {env.scenario_name}. Test Data saved")
