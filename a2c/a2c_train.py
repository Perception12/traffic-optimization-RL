import os
import sys
from a2c_model import A2CAgent
from config import config
from collections import deque
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scenario_parser import parse_arguments
from traffic_environment import TrafficEnv


# Defining the simulation paths
args = parse_arguments()
scenario = args.scenario # choose which scenario

scen_path = f"scenario_{scenario}"
config_path = os.path.abspath(f"../scenarios/{scen_path}/four_way_simulation.sumocfg")
output_path = config.output_paths[scenario-1]

# Initialize traffic environment
env = TrafficEnv(
    config_path=config_path, 
    output_path=output_path,
    scenario_name=config.scenario_names[scenario-1], 
    max_steps=config.max_steps
    )

# Initialize the A2CAgent
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
agent = A2CAgent(input_dim, output_dim)

# Training monitoring
episode_rewards = []
moving_avg_rewards = []
best_avg_reward = -np.inf
window_size = 10  # For moving average
reward_history = deque(maxlen=window_size)

# Training Loop
for episode in range(config.num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action, _ = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        
        agent.update(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
    
    # Store and analyze rewards
    episode_rewards.append(total_reward)
    reward_history.append(total_reward)
    moving_avg = np.mean(reward_history)
    moving_avg_rewards.append(moving_avg)

   # Save best model
    if moving_avg > best_avg_reward:
        best_avg_reward = moving_avg
        agent.save_model(f"models/best_actor_model_scenario_{scenario}.pth", f"models/best_critic_model_scenario_{scenario}.pth")

    # Early stopping
    if episode > window_size and moving_avg >= config.target_reward:
        print(f"Early stopping at episode {episode} - target reward achieved!")
        break
    
    # Progress tracking
    if episode % 20 == 0:
        agent.save_model(f"models/actor_model_{episode}.pth", f"models/critic_model_{episode}.pth")
        print(f"Episode {episode} | Reward: {total_reward:.2f} | Avg Reward (last {window_size}): {moving_avg:.2f}")


# Save final model and plot results
print(f"Training completed for Scenario {scenario}")
print(f"Best Moving Average Reward: {best_avg_reward:.2f}")
agent.save_model(f"models/final_actor_model_scenario_{scenario}.pth", f"models/final_critic_model_scenario_{scenario}.pth")
env.close()

# Plot training progress
plt.figure(figsize=(12, 6))
plt.plot(episode_rewards, label='Episode Reward')
plt.plot(moving_avg_rewards, label=f'Moving Avg ({window_size} episodes)')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Progress A2C')
plt.legend()
plt.savefig(f"results/training_curve_scenario_{scenario}.png")
plt.close()

print(f"Training completed for Scenario {scenario}")