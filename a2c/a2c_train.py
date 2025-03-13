import os
import sys
from a2c_model import A2CAgent
from config import config

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from traffic_environment import TrafficEnv


# Defining the simulation paths
scenario = 3 # choose which scenario

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


agent.save_model(f"models/a2c_actor{scenario}.pth", f"models/a2c_critic{scenario}.pth")


print(f"A2C simulation completed for {env.scenario_name}. Model and results saved.")