Experiment 7: Performance Tracking and Visualization

Objective: Monitor the RL system's performance over time using metrics like cumulative rewards and learning curves.

Implementation Steps:

    Logging Metrics:
        Continuously log cumulative rewards, moving averages, and other relevant metrics.

    Visualization:
        Generate plots to visualize learning curves and performance trends.

    Statistical Analysis:
        Perform correlation analyses to understand relationships between metrics.

Code Integration:

Enhance your logging and create visualization functions. Here's how you can implement performance tracking:

python

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

def plot_learning_curves(rl_rewards, conventional_rewards=None, save_path=None):
"""
Plots the learning curves of RL and conventional systems.

    Args:
        rl_rewards (list of float): Cumulative rewards from the RL system.
        conventional_rewards (list of float, optional): Rewards from conventional methods.
        save_path (str, optional): Path to save the plot.
    """
    plt.figure(figsize=(10,6))
    plt.plot(rl_rewards, label='RL Biomolecular Design System')
    if conventional_rewards:
        plt.plot(conventional_rewards, label='Conventional System')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Learning Curve Comparison')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_moving_average(rewards, window=10, label='RL System', save_path=None):
"""
Plots the moving average of rewards.

    Args:
        rewards (list of float): Rewards to plot.
        window (int): Window size for moving average.
        label (str): Label for the plot.
        save_path (str, optional): Path to save the plot.
    """
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.figure(figsize=(10,6))
    plt.plot(moving_avg, label=f'{label} (MA)')
    plt.xlabel('Episode')
    plt.ylabel('Moving Average Reward')
    plt.title('Moving Average of Rewards')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def correlation_analysis(rl_rewards, conventional_rewards=None):
"""
Performs Pearson correlation analysis on rewards.

    Args:
        rl_rewards (list of float): Rewards from RL system.
        conventional_rewards (list of float, optional): Rewards from conventional system.

    Returns:
        dict: Correlation coefficients.
    """
    correlations = {}
    if conventional_rewards:
        corr, _ = pearsonr(rl_rewards, conventional_rewards)
        correlations['Pearson_Correlation_RL_vs_Conventional'] = corr
    return correlations

# Usage within training

def track_performance(log_data, output_dir):
"""
Tracks and visualizes performance based on logged data.

    Args:
        log_data (list of dict): Logged data entries.
        output_dir (str): Directory to save plots.
    """
    rl_rewards = [entry['Reward'] for entry in log_data if entry['Agent_ID'] == 0]  # Example for Agent 0
    plot_learning_curves(rl_rewards, save_path=os.path.join(output_dir, 'learning_curve.png'))
    plot_moving_average(rl_rewards, window=10, label='RL System', save_path=os.path.join(output_dir, 'moving_average.png'))
    correlations = correlation_analysis(rl_rewards)
    print(f"Correlation Analysis: {correlations}")

Explanation:

    Learning Curves: Visualize how rewards evolve over episodes, indicating learning progress.
    Moving Average: Smooths out fluctuations to reveal underlying trends.
    Correlation Analysis: Helps identify relationships between different performance metrics.
