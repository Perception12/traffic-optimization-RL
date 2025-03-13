from sarsa_model import SARSAAgent  # Import SARSA agent
import os
import sys
from config import config

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from traffic_environment import TrafficEnv


# Defining the simulation paths
config_path = os.path.abspath("../scenarios/scenario_3/four_way_simulation.sumocfg")
output_path = "results/sarsa_heavy_traffic_EW_test_results.csv"

# Initialize traffic environment
env = TrafficEnv(
    config_path=config_path, 
    scenario_name="heavy_traffic_EW", 
    output_path=output_path, 
    max_steps=config.max_steps, 
    gui=True)

# Initialize the A2CAgent
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
agent = SARSAAgent(input_dim, output_dim)  

agent.load_model('models/sarsa_model3.pth')


state = env.reset()
total_reward = 0
done = False
step = 0
    
    
while not done:
    action = agent.select_action(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state
    

    total_reward += reward
    
    step += 1
    print(f"Step {step}: Average Queue Length = {-reward:.2f}")

env.close()

print(f"SARSA simulation completed for {env.scenario_name}. Test Data saved")
