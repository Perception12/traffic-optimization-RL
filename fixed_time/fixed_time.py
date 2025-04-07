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

phase_durations = config.phase_durations[scenario-1]

state = env.reset()
total_reward = 0
done = False
step = 0
time_in_phase = 0
phase_index = 0
moving_avg_rewards = []
window_size = 50  # For moving average
reward_history = deque(maxlen=window_size)

while not done:
    time_in_phase += 1
    if time_in_phase >= phase_durations[phase_index]:  
        phase_index = (phase_index + 1) % output_dim
        time_in_phase = 0 
    
    next_state, reward, done, _ = env.step(phase_index)
    reward_history.append(reward)
    moving_avg = np.mean(reward_history)
    moving_avg_rewards.append(moving_avg)

env.close()

# Plot test rewards
plt.figure(figsize=(12, 6))
plt.plot(moving_avg_rewards, label=f'Step Reward)')
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('Fixed Time Progress')
plt.legend()
plt.savefig(f"results/test_curve_scenario_{scenario}.png")
plt.close()

print(f"Fixed Time testing completed for {env.scenario_name}. Results saved")