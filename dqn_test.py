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

# Load the trained model
agent.load_model("dqn_model1.pth")

#Start SUMO
sumo_cmd = ["sumo", "-c", "four_way_simulation.sumocfg"]
traci.start(sumo_cmd)

# Run the simulation for 1000 steps
max_steps = 1000
step = 0
total_waiting_times = []
action_counts = {i: 0 for i in range(4)}

while step < max_steps:
    state = [traci.edge.getWaitingTime(edge) for edge in edges]  # Get total waiting time
    state = [x / 50 for x in state]
    
    action = agent.select_action(state)

    action_counts[action] += 1

    traci.trafficlight.setRedYellowGreenState("clusterJ3_J4_J6", phases[action])
    traci.simulationStep()

    total_waiting_time = 0
    total_vehicles = 0

    for lane in lanes:
        vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
        total_waiting_time += sum(traci.vehicle.getWaitingTime(veh_id) for veh_id in vehicle_ids)
        total_vehicles += len(vehicle_ids)

    avg_waiting_time = total_waiting_time / total_vehicles if total_vehicles > 0 else 0
    total_waiting_times.append(avg_waiting_time)

    print(f"Step {step}: Average waiting time = {avg_waiting_time:.2f} seconds")
    step += 1

traci.close()

print(action_counts)

# Save the test results
df_test = pd.DataFrame({"Step": range(1, len(total_waiting_times) + 1), "Avg Waiting Time": total_waiting_times})
df_test.to_csv("test_results/dqn_result_scen1.csv", index=False)

print("DQN testing completed. Results saved to dqn_test_results.csv.")


