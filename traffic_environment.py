import traci
import numpy as np
import gym
from gym import spaces
import csv


class TrafficEnv(gym.Env):
    def __init__(self, config_path: str, scenario_name: str, output_path: str, max_steps=500, junction_id="clusterJ10_J11", gui=False):
        super(TrafficEnv, self).__init__()

        self.sumo_mode = 'sumo-gui' if gui else 'sumo'
        self.sumoCmd = [self.sumo_mode, '-c',
                        config_path, '--no-step-log', 'true']
        self.action_space = spaces.Discrete(4)  # 4 possible traffic phases
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(11,))  # queue length, phase, step
        self.junction_id = junction_id
        self.max_steps = max_steps
        self.scenario_name = scenario_name
        self.max_queue_length = 5  # Fixed max queue length for normalization
        self.current_step = 0
        self.data_log = []
        self.phases = ["rGrrrGrr", "GrrrGrrr", "rrrGrrrG", "rrGrrrGr"]
        self.output_path = output_path
        self.max_vehicles_passed = 1
        self.previous_action = None
        self.last_phase_change_step = 0
        self.file = open(self.output_path, 'w', newline="")
        self.writer = csv.writer(self.file)

        self.writer.writerow(['Average Queue length', 'Reward',
                              'Step', 'Total Queue Length', "Avg Waiting Time"])

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
        controlled_lanes = traci.trafficlight.getControlledLanes(
            self.junction_id)

        # Get queue length (normalized)
        queue_length = [max(1, traci.lane.getLastStepHaltingNumber(lane))
                        for lane in controlled_lanes]

        self.max_queue_length = max(5, max(queue_length, default=1))

        state = [x / self.max_queue_length for x in queue_length]

        phase = self.phases.index(traci.trafficlight.getRedYellowGreenState(
            self.junction_id))

        normalized_phase = phase / len(self.phases)

        time_since_change = self.current_step - self.last_phase_change_step
        normalized_time = time_since_change / self.max_steps

        # Normalize step count
        normalized_step = self.current_step / self.max_steps

        state.extend([normalized_phase, normalized_step, normalized_time])
        state = np.array(state, dtype=np.float32)
        return state

    def step(self, action):
        """ Execute an action (change traffic light phase) and return new state, reward, done flag """
        traci.trafficlight.setRedYellowGreenState(
            self.junction_id, self.phases[action])

        # Get the sum of all cars waiting before the intersection for testing purposes
        controlled_lanes = traci.trafficlight.getControlledLanes(
            self.junction_id)

        # Get queue length
        queue_length = [traci.lane.getLastStepHaltingNumber(
            lane) for lane in controlled_lanes]

        queue_length = [x / self.max_queue_length for x in queue_length]

        avg_waiting_time = np.mean([traci.lane.getWaitingTime(
            lane) for lane in controlled_lanes]) if controlled_lanes else 0

        avg_queue_length = np.mean(queue_length)

        total_queue_length = np.sum(
            [traci.lane.getLastStepHaltingNumber(lane) for lane in controlled_lanes])

        # Get the number of vehicles passed
        vehicles_passed = traci.simulation.getArrivedNumber()

        if vehicles_passed > self.max_vehicles_passed:
            self.max_vehicles_passed = vehicles_passed
        # Normalize vehicles_passed
        vehicles_passed = vehicles_passed / self.max_vehicles_passed

        reward = -0.7 * avg_queue_length + 0.3 * \
            vehicles_passed - 0.2 * avg_waiting_time

        # penalize for switching too quickly
        if action != self.previous_action:
            self.last_phase_change_step = self.current_step
            
            
            switch_penalty = 0.05 * np.sqrt(self.current_step - self.last_phase_change_step)
            reward -= switch_penalty
            
            
        print(f"[Step {self.current_step}] Action: {action} | Phase: {self.phases[action]} | Reward: {reward:.3f} | Queue: {avg_queue_length:.2f}")

        self.previous_action = action

        traci.simulationStep()
        self.current_step += 1

        # Get new state
        state = self.get_state()

        # Early Termination
        if total_queue_length > 50:
            reward -= 1.0
            done = True
        else:
            done = self.current_step >= self.max_steps
            # Save to CSV

        # Periodically flush the CSV writer
        if self.current_step % 10 == 0:
            self.file.flush()

        self.writer.writerow([avg_queue_length, reward, self.current_step,
                              total_queue_length, avg_waiting_time])

        # Check if done
        return state, reward, done, {}

    def close(self):
        """ Close the SUMO simulation """
        self.file.close()
        traci.close()
