import traci
import os
from a2c_model import A2CAgent

# Traffic light phases
phases = ["rGrrrGrr", "GrrrGrrr", "rrrGrrrG", "rrGrrrGr"]
edges = ["north_1", "east_1", "west_1", "south_1"]

# Initialize A2C agent
agent = A2CAgent(input_dim=len(edges), output_dim=len(phases))
num_episodes = 50
max_steps = 1500

a2c_waiting_times = []

config_path = os.path.abspath("../scenarios/scenario_1/four_way_simulation.sumocfg")

for episode in range(num_episodes):
    sumo_cmd = ["sumo", "-c", config_path]
    traci.start(sumo_cmd)

    traci.simulationStep()
    state = [traci.edge.getWaitingTime(edge) for edge in edges]  # Per-edge waiting times
    state = [x / 50 for x in state]
    total_reward = 0
    step = 0

    while step < max_steps:
        action, _ = agent.select_action(state)
        traci.trafficlight.setRedYellowGreenState("clusterJ3_J4_J6", phases[action])
        traci.simulationStep()

        # Calculate per-edge waiting time
        per_edge_waiting_times = [traci.edge.getWaitingTime(edge) for edge in edges]
        per_edge_waiting_times = [x / 50 for x in per_edge_waiting_times]
        
        # Calculate reward based on improvement in waiting times
        reward = -sum(per_edge_waiting_times) / len(per_edge_waiting_times)  # Lower is better
        total_reward += reward
        done = traci.simulation.getMinExpectedNumber() == 0

        next_state = per_edge_waiting_times  # Next state is the updated waiting times
        agent.update(state, action, reward, next_state, done)
        state = next_state

        print(f"Episode {episode}, Step {step}, Action: {action}, Reward: {reward:.2f}")

        step += 1
        if done:
            break

    traci.close()

# Save trained models
agent.save_model("a2c_actor3.pth", "a2c_critic3.pth")


print("A2C simulation completed. Results saved to a2c_results.csv and model saved.")