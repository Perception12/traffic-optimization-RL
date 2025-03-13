class Config:
    num_episodes = 50
    max_steps = 1000
    output_paths = ["results/ft_uniform_traffic_train_data.csv", "results/ft_heavy_traffic_NS_train_data.csv", "results/ft_heavy_traffic_WE_train_data.csv"]
    test_output_paths = ["results/ft_uniform_traffic_test_data.csv", "results/ft_heavy_traffic_NS_test_data.csv", "results/ft_heavy_traffic_WE_test_data.csv"]
    scenario_names = ["uniform_traffic", "heavy_traffic_NS", "heavy_traffic_WE"]
    phase_durations = [[10, 10, 10, 10], [10, 10, 10, 20], [10, 20, 10, 10]]
config = Config()