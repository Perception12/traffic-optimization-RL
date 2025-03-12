import traci
import pandas as pd

# Start SUMO
sumoCmd = ["sumo", "-c", "four_way_simulation.sumocfg"]
traci.start(sumoCmd)

# Traffic light phases & durations
phases = ["rGrrrGrr", "GrrrGrrr", "rrrGrrrG", "rrGrrrGr"]
# phase_durations = [10, 10, 10, 10]  # scenario 1
phase_durations = [10, 20, 10, 10]  # scenario 2

step = 0
phase_index = 0  
time_in_phase = 0  

# Include both lanes per edge
lanes = ["north_1_0", "north_1_1", "east_1_0", "east_1_1", "west_1_0", "west_1_1", "south_1_0", "south_1_1"]
fixed_time_waiting_times = []

while step < 1000:  
    traci.trafficlight.setRedYellowGreenState("clusterJ3_J4_J6", phases[phase_index])
    traci.simulationStep()

    total_waiting_time = 0
    total_vehicles = 0
    for lane in lanes:
        vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)  
        total_waiting_time += sum(traci.vehicle.getWaitingTime(veh_id) for veh_id in vehicle_ids)
        total_vehicles += len(vehicle_ids)

    # Compute average waiting time per vehicle
    avg_waiting_time = total_waiting_time / total_vehicles if total_vehicles > 0 else 0
    fixed_time_waiting_times.append(avg_waiting_time)
    print(f"Step {step}: Average waiting time = {avg_waiting_time:.2f} seconds")

    # Phase switching logic
    time_in_phase += 1
    if time_in_phase >= phase_durations[phase_index]:  
        phase_index = (phase_index + 1) % len(phases)  
        time_in_phase = 0  
        
        

    step += 1

traci.close()

# Save to CSV
df = pd.DataFrame({"Step": range(1, len(fixed_time_waiting_times) + 1), "Avg Waiting Time": fixed_time_waiting_times})
df.to_csv("test_results/fixed_time_results_scen2.csv", index=False)

print("Fixed-time simulation completed. Results saved to fixed_time_results.csv.")
