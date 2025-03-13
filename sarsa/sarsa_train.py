from sarsa_model import SARSAAgent  # Import SARSA agent
import os
import sys
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
agent = SARSAAgent(input_dim, output_dim)  


for episode in range(config.num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    step = 0
    
    state = [traci.edge.getWaitingTime(edge) for edge in edges]
    state = [x / 50 for x in state]
    action = agent.select_action(state)
    total_reward = 0
    
    for step in range(max_steps):
        traci.trafficlight.setRedYellowGreenState("clusterJ3_J4_J6", phases[action])
        traci.simulationStep()

        # Calculate per-edge waiting time
        per_edge_waiting_times = [traci.edge.getWaitingTime(edge) for edge in edges]
        per_edge_waiting_times = [x / 50 for x in per_edge_waiting_times]
        
        # Calculate reward based on improvement in waiting times
        reward = -sum(per_edge_waiting_times) / len(per_edge_waiting_times)  # Lower is better
        total_reward += reward

        next_state = per_edge_waiting_times
        next_action = agent.select_action(next_state)

        done = traci.simulation.getMinExpectedNumber() == 0

        agent.update(state, action, reward, next_state, next_action, done)

        state, action = next_state, next_action
        total_reward += reward

        print(f"Episode {episode}, Step {step}, Action: {action}, Reward: {reward}")

        if done:
            break

    traci.close()

agent.save_model("sarsa_model3.pth")

print("SARSA simulation completed. Model saved")
