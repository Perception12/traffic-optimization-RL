import traci
import pandas as pd
import numpy as np
import csv

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
junction_id = "clusterJ3_J4_J6"
max_steps = 1000
max_queue_length = 5
output_path = ""

# Open CSV file for logs
with open(output_path, 'w', newline="") as file:
    writer = csv.writer(file)
    writer.writerow(['Queue length', 'Speed', 'Phase', 'Step', 'Action', 'Reward', 'Total Queue Length'])


while step < max_steps:  
    traci.trafficlight.setRedYellowGreenState(junction_id, phases[phase_index])
    traci.simulationStep()
    
    controlled_lanes = traci.trafficlight.getControlledLanes(junction_id)

    # Phase switching logic
    time_in_phase += 1
    if time_in_phase >= phase_durations[phase_index]:  
        phase_index = (phase_index + 1) % len(phases)  
        time_in_phase = 0  
        
    # Get queue length (normalized)
    queue_length = np.mean([traci.lane.getLastStepHaltingNumber(lane) for lane in controlled_lanes])

    # Get average vehicle speed (normalized)
    vehicle_speeds = [traci.vehicle.getSpeed(veh) for veh in traci.vehicle.getIDList()]
    speed = np.mean(vehicle_speeds) if vehicle_speeds else 0  # Avoid NaN
    phase = phase_index / (len(phases)-1)
    # Normalize step count
    step_norm = step / max_steps
    total_queue_length = np.sum([traci.lane.getLastStepHaltingNumber(lane) for lane in controlled_lanes])
    
    reward = -queue_length/max_queue_length
    print(f"Step {step}: Mean Queue Length = {queue_length:.2f}")
    
    state = np.array([queue_length, speed, phase, step_norm])
    # Save to CSV
    with open(output_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([*state, phase_index, reward, total_queue_length])
        
        
    step += 1

traci.close()

print("Fixed-time simulation completed. Results saved to fixed_time_results.csv.")
