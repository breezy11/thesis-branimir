import pandas as pd
from matplotlib import pyplot as plt
import os
import numpy as np
import tradingeconomics as te
import json

te.login('86A02B2788834D4:AD2DEAA48B0144C')

# Function to plot and save transaction data for each year
def plot_transactions_per_year(df_test, model_name, short_ticks, long_ticks):
	"""
	    Plots the transactions per year for a given model.

	    Parameters:
	    df_test (pd.DataFrame): Test dataset containing the price data.
	    model_name (str): Name of the model.
	    short_ticks (list): Indices where short positions are taken.
	    long_ticks (list): Indices where long positions are taken.
	"""

	# Calculate position changes
	position_series = pd.Series(index=df_test.index, data=np.nan)
	position_series.iloc[short_ticks] = 0
	position_series.iloc[long_ticks] = 1
	position_series.dropna(inplace=True)
	position_diff = position_series.diff()
	transitions_df = pd.DataFrame(position_series[(position_diff == 1) | (position_diff == -1)])

	# Unique years in the transitions data
	years = transitions_df.index.year.unique()

	# Directory setup
	base_dir = "Training/Results"
	save_dir = f"{base_dir}/{model_name}/transaction_plots"
	if not os.path.exists(f"{base_dir}/{model_name}"):
		os.makedirs(f"{base_dir}/{model_name}")
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	# Plot transactions per year
	for year in years:
		year_transitions = transitions_df[transitions_df.index.year == year]
		close_prices = df_test[(df_test.index.year == year)]['Price']

		plt.figure(figsize=(24, 12))
		plt.plot(close_prices.index, close_prices, label='Price', color='blue')

		longs = year_transitions[year_transitions[0] == 1.0]
		shorts = year_transitions[year_transitions[0] == 0.0]

		plt.scatter(longs.index, close_prices.loc[longs.index], color='green', marker='o', s=50)
		plt.scatter(shorts.index, close_prices.loc[shorts.index], color='red', marker='x', s=50)

		plt.xlabel('Date')
		plt.ylabel('EUR/USD')
		plt.title(f'Model ({model_name}) transactions - {year}')
		plt.grid(True)

		# Manually adding legend with specified handles and labels
		plt.legend(handles=[
			plt.Line2D([], [], color='green', marker='o', linestyle='None', markersize=10),
			plt.Line2D([], [], color='red', marker='x', linestyle='None', markersize=10)
		], labels=['Buy', 'Sell'])

		plt.savefig(f"{save_dir}/{year}.png")
		plt.close()


def plot_model_loss(model_losses, model_name):
	"""
	    Plot the moving average of model losses over time.

	    Parameters:
	    model_losses (list or np.array): List or numpy array of model loss values over time.
	    model_name (str): Name of the model for plot title and save directory.

	    Returns:
	    None
	"""

	save_dir = f"Training/Results/{model_name}"
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	model_losses = np.array(model_losses)
	window_length = int(len(model_losses) / 100)
	moving_average = np.convolve(model_losses, np.ones(window_length) / window_length, mode='valid')

	timesteps = range(1, len(moving_average) + 1)

	# Plot
	plt.figure(figsize=(24, 12))
	plt.plot(timesteps, moving_average, marker='o', linestyle='-', color='b', label='Model Loss')
	plt.title(f'Model ({model_name}) Train Loss')
	plt.xlabel('Timesteps')
	plt.ylabel('Model Loss')
	plt.grid(True)
	plt.legend()
	plt.tight_layout()

	plt.savefig(f'{save_dir}/model_loss_plot.png')
	plt.close()

def plot_train_reward(env_train, model_name):
	"""
	    Plot the moving average and cumulative rewards during training.

	    Parameters:
	    env_train (object): Environment object containing training data and rewards.
	    model_name (str): Name of the model for plot title and save directory.

	    Returns:
	    None
	"""

	save_dir = f"Training/Results/{model_name}"
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	rewards_list = list(env_train.list_rewards.values())
	rewards_array = np.array(rewards_list)
	train_rewards = rewards_array[rewards_array != 0]

	# Plot moving average of rewards
	window_length = int(len(train_rewards) / 5)
	moving_average = np.convolve(train_rewards, np.ones(window_length) / window_length, mode='valid')
	timesteps = range(1, len(moving_average) + 1)

	plt.figure(figsize=(24, 12))
	plt.plot(timesteps, moving_average, marker='o', linestyle='-', color='g', label='Reward')
	plt.title(f'Model ({model_name}) Train Reward')
	plt.ylabel('Reward')
	plt.xlabel('Transactions')
	plt.grid(True)
	plt.legend()
	plt.tight_layout()
	plt.savefig(f'{save_dir}/train_reward_plot.png')
	plt.close()

	# Plot cumulative rewards
	cumulative_rewards = np.cumsum(train_rewards)
	timesteps = range(1, len(cumulative_rewards) + 1)

	plt.figure(figsize=(24, 12))
	plt.plot(timesteps, cumulative_rewards, marker='o', linestyle='-', color='g', label='Reward')
	plt.title(f'Model ({model_name}) Train Cummulative Reward')
	plt.ylabel('Reward')
	plt.xlabel('Transactions')
	plt.grid(True)
	plt.legend()
	plt.tight_layout()
	plt.savefig(f'{save_dir}/train_cummulative_reward_plot.png')
	plt.close()


def plot_test_reward(env_test, df_test, model_name):
	"""
	    Plot the monthly and cumulative rewards during testing.

	    Parameters:
	    env_test (object): Environment object containing testing data and rewards.
	    df_test (pd.DataFrame): DataFrame containing testing data, including date index and 'Price'.
	    model_name (str): Name of the model for plot title and save directory.

	    Returns:
	    None
	"""

	save_dir = f"Training/Results/{model_name}"
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	rewards_list = list(env_test.list_rewards.values())
	rewards_array = np.array(rewards_list)
	test_rewards = rewards_array[rewards_array != 0]

	index_discrepancy = df_test.shape[0] - len(rewards_list)
	df_monthly_rewards = pd.DataFrame(index=df_test.iloc[index_discrepancy:, ].index, columns=['Rewards'],
									  data=rewards_list)
	monthly_rewards = df_monthly_rewards.resample('M').sum()

	# Plot monthly rewards
	plt.figure(figsize=(24, 12))
	plt.plot(monthly_rewards.index, monthly_rewards['Rewards'], marker='o', linestyle='-', color='g')
	plt.title(f'Model ({model_name}) Test Monthly Reward')
	plt.xlabel('Date')
	plt.ylabel('Monthly Reward')
	plt.grid(True)
	plt.tight_layout()

	plt.savefig(f'{save_dir}/test_monthly_reward_plot.png')
	plt.close()

	# Plot cumulative rewards
	cumulative_rewards = np.cumsum(test_rewards)
	timesteps = range(1, len(cumulative_rewards) + 1)

	plt.figure(figsize=(24, 12))
	plt.plot(timesteps, cumulative_rewards, linestyle='-', color='g', label='Reward')
	plt.title(f'Model ({model_name}) Test Cummulative Reward')
	plt.ylabel('Reward')
	plt.xlabel('Transactions')
	plt.grid(True)
	plt.legend()
	plt.tight_layout()
	plt.savefig(f'{save_dir}/test_cummulative_reward_plot.png')
	plt.close()

def calculate_store_metrics(env_test, df_test, losses, model_name):

	# Calculate mean loss of the last (1/10) * n loss values
	index_separation = int((1/10) * len(losses))
	mean_loss = np.mean(losses[-index_separation:])

	# Get total reward from the testing environment
	results = env_test._get_info()
	total_reward = results['total_reward']

	# Calculate Sharpe ratio
	monthly_risk_free_rate = (1 + 0.03) ** (1 / 12) - 1
	rewards_list = list(env_test.list_rewards.values())
	rewards_array = np.array(rewards_list)
	test_rewards = rewards_array[rewards_array != 0]
	index_discrepancy = df_test.shape[0] - len(rewards_list)
	df_monthly_rewards = pd.DataFrame(index=df_test.iloc[index_discrepancy:, ].index, columns=['Rewards'],
									  data=rewards_list)
	monthly_rewards = df_monthly_rewards.resample('M').sum()
	R_p = monthly_rewards['Rewards'].mean()
	R_std = monthly_rewards['Rewards'].std()

	sharpe_ratio = (R_p - monthly_risk_free_rate) / R_std

	# Calculate maximum drawdown
	cumulative_returns = np.cumsum(test_rewards)
	peak = cumulative_returns[0]
	max_drawdown = 0.0
	for i in range(1, len(cumulative_returns)):
		if cumulative_returns[i] > peak:
			peak = cumulative_returns[i]
		else:
			drawdown = (peak - cumulative_returns[i]) / peak
			if drawdown > max_drawdown:
				max_drawdown = drawdown

	# Calculate volatility
	average_return = monthly_rewards['Rewards'].mean()
	deviations = monthly_rewards - average_return
	squared_deviations = deviations ** 2
	mean_squared_deviations = squared_deviations['Rewards'].mean()
	volatility = np.sqrt(mean_squared_deviations)

	# Calculate beta
	df = te.getHistorical(symbol='SPX:IND', output_type='df', initDate='2010-01-01', endDate='2017-01-01')
	df.set_index('Date', inplace=True)
	df.index = pd.to_datetime(df.index, format='%d/%m/%Y')
	df = pd.DataFrame(df['Close'])
	sp_500_monthly_returns = df['Close'].resample('M').last().pct_change().dropna()

	monthly_rewards = monthly_rewards[1:]

	covariance = np.cov(monthly_rewards['Rewards'], sp_500_monthly_returns)[0, 1]
	variance_sp500 = np.var(sp_500_monthly_returns)

	beta = covariance / variance_sp500

	# Get return of the strategy
	return_strategy = results['total_profit']

	final_results = {'Mean loss': mean_loss,
					 'Total reward': total_reward.round(3),
					 'Total return': return_strategy.round(3),
					 'Sharpe ratio': sharpe_ratio.round(3),
					 'Maximum drawdown': max_drawdown.round(3),
					 'Volatility': volatility.round(3),
					 'Beta': beta.round(3)}

	save_dir = f"Training/Results/{model_name}"
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	with open(f'{save_dir}/results.json', 'w') as f:
		json.dump(final_results, f, indent=4)

	return final_results

def calculate_metrics_create_plots(env_train, env_test, df_test, losses, short_ticks, long_ticks, model_name):
	final_results = calculate_store_metrics(env_test, df_test, losses, model_name)
	plot_transactions_per_year(df_test, model_name, short_ticks, long_ticks)
	plot_model_loss(losses, model_name)
	plot_train_reward(env_train, model_name)
	plot_test_reward(env_test, df_test, model_name)

	return final_results