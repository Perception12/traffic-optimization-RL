import traci
import numpy as np
import gym
from gym import spaces
import csv

class TrafficEnv(gym.Env):
    def __init__(self, config_path: str, scenario_name: str, output_path:str, max_steps=500, junction_id="clusterJ3_J4_J6", gui=False):
        super(TrafficEnv, self).__init__()
        
        self.sumo_mode = 'sumo-gui' if gui else 'sumo'
        self.sumoCmd = [self.sumo_mode, '-c', config_path, '--no-step-log', 'true']
        self.action_space = spaces.Discrete(4)  # 4 possible traffic phases
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,))  # queue length, phase, step
        self.junction_id = junction_id
        self.max_steps = max_steps
        self.scenario_name = scenario_name
        self.max_queue_length = 5  # Fixed max queue length for normalization
        self.current_step = 0
        self.data_log = []
        self.phases = ["rGrrrGrr", "GrrrGrrr", "rrrGrrrG", "rrGrrrGr"]
        self.output_path = output_path
        
        # Open CSV file for logs
        with open(self.output_path, 'w', newline="") as file:
            writer = csv.writer(file)
            writer.writerow(['Average Queue length', 'Phase', 'Step', 'Action', 'Reward', 'Total Queue Length'])
        
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
        queue_length = [traci.lane.getLastStepHaltingNumber(lane) for lane in controlled_lanes]
            
        state = [x / self.max_queue_length for x in queue_length]
        
        # Normalize traffic light phase
        phase = self.phases.index(traci.trafficlight.getRedYellowGreenState(self.junction_id)) / (len(self.phases) - 1)

        # Normalize step count
        step_norm = self.current_step / self.max_steps
        
        
        state.extend([phase, step_norm])
        state = np.array(state, dtype=np.float32)
        return state
    
    def step(self, action):
        # Get the sum of all cars waiting before the intersection for testing purposes
        controlled_lanes = traci.trafficlight.getControlledLanes(self.junction_id)
        
        avg_queue_length = np.mean([traci.lane.getLastStepHaltingNumber(lane) for lane in controlled_lanes])
        avg_queue_length = avg_queue_length / self.max_queue_length
        
        total_queue_length = np.sum([traci.lane.getLastStepHaltingNumber(lane) for lane in controlled_lanes])
        vehicles_passed = traci.simulation.getArrivedNumber()
        reward = -avg_queue_length + 0.5 * vehicles_passed
            
        """ Execute an action (change traffic light phase) and return new state, reward, done flag """
        traci.trafficlight.setRedYellowGreenState(self.junction_id, self.phases[action])
    
        traci.simulationStep()
        self.current_step += 1

        # Get new state
        state = self.get_state()

        # Save to CSV
        with open(self.output_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([avg_queue_length, state[-2], state[-1], action, reward, total_queue_length])

        # Check if done
        done = self.current_step >= self.max_steps
        return state, reward, done, {}
    
    def close(self):
        """ Close the SUMO simulation """
        traci.close()
