class Config:
    num_episodes = 50
    max_steps = 1000
    learning_rate = 0.001
    output_paths = ["results/a2cuniform_traffic_train_data.csv", "results/a2cheavy_traffic_NS_train_data.csv", "results/a2cheavy_traffic_WE_train_data.csv"]
    test_output_paths = ["results/a2cuniform_traffic_test_data.csv", "results/a2cheavy_traffic_NS_test_data.csv", "results/a2cheavy_traffic_WE_test_data.csv"]
    scenario_names = ["uniform_traffic", "heavy_traffic_NS", "heavy_traffic_WE"]
config = Config()