import os
import sys
from a2c_model import A2CAgent
from config import config

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from traffic_environment import TrafficEnv


# Defining the simulation paths
config_path = os.path.abspath("../scenarios/scenario_2/four_way_simulation.sumocfg")
output_path = "traffic_data.csv"

# Initialize traffic environment
env = TrafficEnv(config_path, scenario_name="a2c_heavy_NS", max_steps=config.max_steps)

# Initialize the A2CAgent
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
agent = A2CAgent(input_dim, output_dim)

for episode in range(config.num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    step = 0
    
    while not done:
        action, _ = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        
        agent.update(state, action, reward, next_state, done)
        state = next_state

        total_reward += reward
        step += 1
        print(f"Episode {episode}, Step {step}, Action: {action}, Reward: {reward:.2f}")

    print(f"Episode {episode} finished with total reward: {total_reward}")

env.close()

# Save trained models
agent.save_model("models/a2c_actor2.pth", "models/a2c_critic2.pth")


print("A2C simulation completed. Results saved to a2c_results.csv and model saved.")