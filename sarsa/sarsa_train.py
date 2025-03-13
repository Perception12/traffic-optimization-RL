from sarsa_model import SARSAAgent  # Import SARSA agent
import os
import sys
from config import config

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from traffic_environment import TrafficEnv


# Defining the simulation paths
config_path = os.path.abspath("../scenarios/scenario_3/four_way_simulation.sumocfg")
output_path = "results/heavy_traffic_EW_train_data.csv"

# Initialize traffic environment
env = TrafficEnv(
    config_path=config_path, 
    scenario_name="heavy_traffic_EW", 
    output_path= output_path, 
    max_steps=config.max_steps
    )

# Initialize the A2CAgent
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
agent = SARSAAgent(input_dim, output_dim)  


for episode in range(config.num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    step = 0
    
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)

        next_action = agent.select_action(next_state)

        agent.update(state, action, reward, next_state, next_action, done)

        state, action = next_state, next_action
        total_reward += reward
        
        step += 1
        print(f"Episode {episode}, Step {step}, Action: {action}, Reward: {reward}")

env.close()

agent.save_model("models/sarsa_model3.pth")

print(f"SARSA training completed for {env.scenario_name}. Model saved")
