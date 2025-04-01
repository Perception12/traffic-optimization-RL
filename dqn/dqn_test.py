from dqn_model import DQNAgent
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from traffic_environment import TrafficEnv
from config import config

scenario = 1
# Defining the simulation paths
config_path = os.path.abspath(f"../scenarios/scenario_{scenario}/four_way_simulation.sumocfg")
output_path = config.test_output_paths[scenario-1]

# Initialize traffic environment
env = TrafficEnv(
    config_path=config_path,
    output_path=output_path,
    scenario_name=config.scenario_names[scenario-1],
    max_steps=config.max_steps,
    gui=True)

# Intialize DQN Agent
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
agent = DQNAgent(input_dim, output_dim, epsilon=0.01)  # Set epsilon to a low value for testing

# Load the trained model
agent.load_model(f"models/best_dqn_model_scenario_{scenario}.pth")

# Run the simulation for 1000 steps
state = env.reset()
total_reward = 0
done = False

while not done:
    action = agent.select_action(state) # Choose action from policy
    next_state, reward, done, _ = env.step(action) # Step through environment
    state = next_state

    total_reward += reward
env.close()


print("DQN testing completed. Results saved to test data saved")


