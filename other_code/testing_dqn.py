# imports
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from gym_anytrading.envs import StocksEnv
import pandas as pd
import random
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from helper_code.metrics_calculation import calculate_metrics_create_plots
import json

# Setting up the random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
set_random_seed(seed)

save_path = os.path.join('Training', 'Models')

# Function to prepare and scale the data
def load_and_prepare_data():
    df_train = pd.read_csv('data/df_train.csv', index_col=0)
    df_train.index = pd.to_datetime(df_train.index)

    df_test = pd.read_csv('data/df_test.csv', index_col=0)
    df_test.index = pd.to_datetime(df_test.index)

    # Removing duplicates in index (if any)
    df_train = df_train[~df_train.index.duplicated(keep='first')]
    df_test = df_test[~df_test.index.duplicated(keep='first')]

    scaler = MinMaxScaler(feature_range=(0, 1))

    df_train = pd.DataFrame(scaler.fit_transform(df_train), columns=df_train.columns, index=df_train.index)

    df_test = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns, index=df_test.index)

    return df_train, df_test

# Define custom environment
def add_signals(env, features):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'Price'].to_numpy()[start:end]
    signal_features = env.df.loc[:, features].to_numpy()[start:end]
    return prices, signal_features

class MyCustomEnv(StocksEnv):
    def __init__(self, df, window_size, frame_bound, features, render_mode=None):
        self.features = features  # Store features as an instance attribute
        self.frame_bound = frame_bound
        super().__init__(df, window_size, frame_bound, render_mode)

    def _process_data(self):
        return add_signals(self, self.features)

# Function to train and evaluate the model
def train_and_evaluate_model(df_train, df_test, features, model_name, n_runs=5, train_timesteps=500000):
    all_results = []

    for i in range(n_runs):
        print(f"Run {i+1}/{n_runs}")

        # Create training and testing environments
        env_train = MyCustomEnv(df=df_train, window_size=12, frame_bound=(12, 440473), features=features)
        env_test = MyCustomEnv(df=df_test, window_size=12, frame_bound=(12, 224933), features=features)

        # Wrap environments
        env_wrapped = DummyVecEnv([lambda: env_train])

        # Initialize DQN model
        model = DQN("MlpPolicy", env_wrapped, verbose=2)

        # Train the model
        model.learn(total_timesteps=train_timesteps)

        # Test the model
        obs = env_test.reset()[0]
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, extra, done, info = env_test.step(action)
            if done:
                print("info", info)
                break

        short_ticks, long_ticks = env_test.render_all()

        # Store the results and calculate metrics
        final_results = calculate_metrics_create_plots(env_train, env_test, df_test, model.losses, short_ticks, long_ticks, f'{model_name}_{i}')
        all_results.append(final_results)

    return all_results

# Main execution block
if __name__ == "__main__":
    # Load and prepare data
    df_train, df_test = load_and_prepare_data()

    # Features to use
    features = ['bollinger_hband', 'bollinger_lband', 'ulcer_index', 'sma',
       'ema', 'wma', 'macd', 'trix', 'dpo', 'kst', 'stc', 'aroon_down',
       'aroon_up', 'rsi', 'stochrsi', 'tsi', 'kama', 'roc', 'ppo',
       'ppo_signal']

    model_name = 'test'

    # Run training and evaluation 5 times
    all_results = train_and_evaluate_model(df_train, df_test, features, model_name, n_runs=5)

    result = {i: all_results[i] for i in range(len(all_results))}

    final_results = {}
    num_entries = len(result)

    for key in result[0].keys():
        total = sum(d[key] for d in result.values())
        final_results[key] = total / num_entries

    result['final_results'] = final_results

    save_dir = f"Training/Results/{model_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(f'{save_dir}/combined_results.json', 'w') as f:
        json.dump(result, f, indent=4)


