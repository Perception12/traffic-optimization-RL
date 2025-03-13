import traci
import numpy as np
import gym
from gym import spaces
import csv

class TrafficEnv(gym.Env):
    def __init__(self, config_path: str, scenario_name: str, max_steps=500, junction_id="clusterJ3_J4_J6"):
        super(TrafficEnv, self).__init__()
        
        self.sumoCmd = ['sumo', '-c', config_path, '--no-step-log', 'true']
        self.action_space = spaces.Discrete(4)  # 4 possible traffic phases
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,))  # queue length, speed, phase, step
        self.junction_id = junction_id
        self.max_steps = max_steps
        self.scenario_name = scenario_name
        self.max_queue_length = 15  # Fixed max queue length for normalization
        self.current_step = 0
        self.data_log = []
        self.phases = ["rGrrrGrr", "GrrrGrrr", "rrrGrrrG", "rrGrrrGr"]
        
        self.output_path = f"{scenario_name}_traffic_data.csv"
        # Open CSV file for logs
        with open(self.output_path, 'w', newline="") as file:
            writer = csv.writer(file)
            writer.writerow(['Queue length', 'Speed', 'Phase', 'Step', 'Action', 'Reward'])
        
    def reset(self):
        """ Reset the environment and return initial state """
        try:
            traci.start(self.sumoCmd)
        except traci.exceptions.TraCIException:
            traci.close()
            traci.start(self.sumoCmd)

        self.current_step = 0
        return self.get_state()
    
    def get_state(self):
        """ Get normalized state from SUMO """
        controlled_lanes = traci.trafficlight.getControlledLanes(self.junction_id)
        
        # Get queue length (normalized)
        queue_length = np.mean([traci.lane.getLastStepHaltingNumber(lane) for lane in controlled_lanes])
        queue_normalized = min(queue_length / self.max_queue_length, 1.0)  # Cap at 1.0
        
        # Get average vehicle speed (normalized)
        vehicle_speeds = [traci.vehicle.getSpeed(veh) for veh in traci.vehicle.getIDList()]
        speed = np.mean(vehicle_speeds) if vehicle_speeds else 0  # Avoid NaN
        
        # Normalize traffic light phase
        phase = self.phases.index(traci.trafficlight.getRedYellowGreenState(self.junction_id)) / (len(self.phases) - 1)

        # Normalize step count
        step_norm = self.current_step / self.max_steps

        state = np.array([queue_normalized, speed, phase, step_norm], dtype=np.float32)
        return state
    
    def step(self, action):
        """ Execute an action (change traffic light phase) and return new state, reward, done flag """
        traci.trafficlight.setRedYellowGreenState(self.junction_id, self.phases[action])
    
        traci.simulationStep()
        self.current_step += 1

        # Get new state
        state = self.get_state()

        # Reward: minimize queue length (negative reward for higher queue)
        reward = -state[0]  

        # Save step data
        self.data_log.append([*state, action, reward])

        # Save to CSV
        with open(self.output_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([*state, action, reward])

        # Check if done
        done = self.current_step >= self.max_steps
        return state, reward, done, {}
    
    def close(self):
        """ Close the SUMO simulation """
        traci.close()
