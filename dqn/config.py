class Config:
    num_episodes = 50
    batch_size = 32
    max_steps = 1000
    learning_rate = 0.001
    output_paths = ["results/uniform_traffic_NS_train_data.csv", "results/heavy_traffic_train_data.csv", "results/heavy_traffic_WE_train_data.csv"]
    test_output_paths = ["results/uniform_traffic_NS_test_data.csv", "results/heavy_traffic_test_data.csv", "results/heavy_traffic_WE_test_data.csv"]
    scenario_names = ["uniform_traffic", "heavy_traffic_NS", "heavy_traffic_WE"]

config = Config()