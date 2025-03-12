from sarsa_model import SARSAAgent  # Import SARSA agent
import traci
import pandas as pd

phases = ["rGrrrGrr", "GrrrGrrr", "rrrGrrrG", "rrGrrrGr"]
edges = ["north_1", "east_1", "west_1", "south_1"]
lanes = ["north_1_0", "north_1_1", "east_1_0", "east_1_1", "west_1_0", "west_1_1", "south_1_0", "south_1_1"]

agent = SARSAAgent(input_dim=len(edges), output_dim=4)  
num_episodes = 50
max_steps = 1500


for episode in range(num_episodes):
    traci.start(["sumo", "-c", "four_way_simulation.sumocfg"])
    traci.simulationStep()
    
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
