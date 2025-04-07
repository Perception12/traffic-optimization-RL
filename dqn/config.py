class Config:
    num_episodes = 500
    batch_size = 32
    max_steps = 3000
    target_reward= -5.0
    learning_rate = 0.001
    output_paths = ["results/dqn_uniform_traffic_train_data.csv", "results/dqn_heavy_traffic_NS_train_data.csv", "results/dqn_heavy_traffic_WE_train_data.csv"]
    test_output_paths = ["results/dqn_uniform_traffic_test_data.csv", "results/dqn_heavy_traffic_NS_test_data.csv", "results/dqn_heavy_traffic_WE_test_data.csv"]
    scenario_names = ["uniform_traffic", "heavy_traffic_NS", "heavy_traffic_WE"]

config = Config()