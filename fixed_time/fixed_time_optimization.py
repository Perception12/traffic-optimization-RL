import traci
import pandas as pd
import itertools

# Traffic light phases
phases = ["rGrrrGrr", "GrrrGrrr", "rrrGrrrG", "rrGrrrGr"]

# Include both lanes per edge
lanes = ["north_1_0", "north_1_1", "east_1_0", "east_1_1", "west_1_0", "west_1_1", "south_1_0", "south_1_1"]

def run_simulation(phase_durations):
    sumoCmd = ["sumo", "-c", "four_way_simulation.sumocfg"]
    traci.start(sumoCmd)

    step = 0
    phase_index = 0
    time_in_phase = 0
    fixed_time_waiting_times = []

    while step < 500:
        traci.trafficlight.setRedYellowGreenState("clusterJ3_J4_J6", phases[phase_index])
        traci.simulationStep()

        total_waiting_time = 0
        total_vehicles = 0
        for lane in lanes:
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
            total_waiting_time += sum(traci.vehicle.getWaitingTime(veh_id) for veh_id in vehicle_ids)
            total_vehicles += len(vehicle_ids)

        avg_waiting_time = total_waiting_time / total_vehicles if total_vehicles > 0 else 0
        fixed_time_waiting_times.append(avg_waiting_time)

        time_in_phase += 1
        if time_in_phase >= phase_durations[phase_index]:
            phase_index = (phase_index + 1) % len(phases)
            time_in_phase = 0

        step += 1

    traci.close()
    return sum(fixed_time_waiting_times) / len(fixed_time_waiting_times)

# Define the range of durations to test for each phase
duration_range = range(10, 41, 10)  # Example: 10 to 40 seconds in steps of 10

# Generate all combinations of phase durations
all_combinations = list(itertools.product(duration_range, repeat=len(phases)))

# Run the simulation for each combination and store the results
results = []
for combination in all_combinations:
    avg_waiting_time = run_simulation(combination)
    results.append((combination, avg_waiting_time))
    print(f"Combination: {combination}, Average Waiting Time: {avg_waiting_time:.2f} seconds")

# Find the combination with the lowest average waiting time
best_combination, best_waiting_time = min(results, key=lambda x: x[1])
print(f"Best Combination: {best_combination}, Best Average Waiting Time: {best_waiting_time:.2f} seconds")

# Save the results to a CSV file
df_results = pd.DataFrame(results, columns=["Phase Durations", "Average Waiting Time"])
df_results.to_csv("fixed_time_optimization_results.csv", index=False)

print("Optimization completed. Results saved to fixed_time_optimization_results.csv.")
