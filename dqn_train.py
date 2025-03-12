from dqn_model import DQNAgent
import traci
import pandas as pd

# Traffic light phases
phases = ["rGrrrGrr", "GrrrGrrr", "rrrGrrrG", "rrGrrrGr"]

# Include both lanes per edge
edges = ["north_1", "east_1", "west_1", "south_1"]

lanes = ["north_1_0", "north_1_1", "east_1_0", "east_1_1", "west_1_0", "west_1_1", "south_1_0", "south_1_1"]

# Initialize DQN agent
agent = DQNAgent(input_dim=len(edges), output_dim=len(phases))  
num_episodes = 50
batch_size = 32
max_steps = 1500  # Define the maximum number of steps per episode



for episode in range(num_episodes):
    # Start SUMO
    sumo_cmd = ["sumo", "-c", "four_way_simulation.sumocfg"]
    traci.start(sumo_cmd)
    
    traci.simulationStep()
    state = [traci.edge.getWaitingTime(edge) for edge in edges]  # Get total waiting time
    state = [x / 50 for x in state]
    total_reward = 0
    step = 0

    while step < max_steps:
        action = agent.select_action(state)
        traci.trafficlight.setRedYellowGreenState("clusterJ3_J4_J6", phases[action])
        traci.simulationStep()

        
        # Calculate per-edge waiting time
        per_edge_waiting_times = [traci.edge.getWaitingTime(edge) for edge in edges]
        per_edge_waiting_times = [x / 50 for x in per_edge_waiting_times]
        
        # Calculate reward based on improvement in waiting times
        reward = -sum(per_edge_waiting_times) / len(per_edge_waiting_times)  # Lower is better
        total_reward += reward

        done = traci.simulation.getMinExpectedNumber() == 0  # Stop when no vehicles remain
        if done:
            break

        # Store transition in memory
        next_state = per_edge_waiting_times
        agent.memory.push((state, action, reward, next_state, done))
        state = next_state
        
        print(f"Episode {episode}, Step {step}, Action: {action}")
        print(f"State: {state}")
        print(f"Next State: {next_state}")
        print(f"Reward: {reward}")

        agent.update(batch_size)
        step += 1

    traci.close()  # Close the current simulation

# Save the trained model
agent.save_model("dqn_model3.pth")

print("DQN simulation completed. Model saved to dqn_model.pth.")
