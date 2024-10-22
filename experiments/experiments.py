import os
import json
from rdkit import Chem
from rdkit.Chem import Descriptors
from transform_and_path import PATHWAY_SCORING_FUNCTIONS
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
def plot_learning_curves(rl_rewards, conventional_rewards=None, output_dir=None):
    plt.figure(figsize=(10,6))
    plt.plot(rl_rewards, label='RL Biomolecular Design System')
    if conventional_rewards:
        plt.plot(conventional_rewards, label='Conventional System')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Learning Curve Comparison')
    plt.legend()
    if output_dir:
        plt.savefig(output_dir)
    plt.close()  # Close the figure instead of showing it

def plot_moving_average(rewards, window=10, label='RL System', output_dir=None):
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.figure(figsize=(10,6))
    plt.plot(moving_avg, label=f'{label} (MA)')
    plt.xlabel('Episode')
    plt.ylabel('Moving Average Reward')
    plt.title('Moving Average of Rewards')
    plt.legend()
    if output_dir:
        plt.savefig(output_dir)
    plt.close()  # Close the figure instead of showing it

def track_performance(log_data, output_dir):
    rl_rewards = [entry['Reward'] for entry in log_data if entry['Agent_ID'] == 0]  # Example for Agent 0
    plot_learning_curves(rl_rewards, output_dir=os.path.join(output_dir, 'learning_curve.png'))
    plot_moving_average(rl_rewards, window=10, label='RL System', output_dir=os.path.join(output_dir, 'moving_average.png'))
    correlations = correlation_analysis(rl_rewards)
    with open(os.path.join(output_dir, 'correlation_analysis.json'), 'w') as f:
        json.dump(correlations, f)

def correlation_analysis(rl_rewards, conventional_rewards=None):
    correlations = {}
    if conventional_rewards:
        corr, _ = pearsonr(rl_rewards, conventional_rewards)
        correlations['Pearson_Correlation_RL_vs_Conventional'] = corr
    return correlations






def assess_polypharmacology_synergy(molecules, agent_objectives):
    synergy_scores = []

    for mol in molecules:
        scores = []
        for objective in agent_objectives:
            if objective in PATHWAY_SCORING_FUNCTIONS:
                score = PATHWAY_SCORING_FUNCTIONS[objective](mol)
                if score is not None:
                    scores.append(score)
        
        # Define synergy as the average of pathway scores
        synergy = np.mean(scores) if scores else 0.0
        synergy_scores.append(synergy)

    average_synergy = np.mean(synergy_scores) if synergy_scores else 0.0
    return average_synergy



def validate_polypharmacology(agent, env, n_episodes=10):
    """
    Validates polypharmacology synergy of the agent.

    Args:
        agent: The agent to validate.
        env: The environment associated with the agent.
        n_episodes (int): Number of validation episodes.

    Returns:
        dict: Contains average synergy score.
    """
    total_synergy = 0
    total_molecules = 0

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            state, reward, done, _ = env.step(action)

        mol = env.current_mol
        if mol and Chem.MolToSmiles(mol) != '':
            synergy = assess_polypharmacology_synergy([mol], agent.objectives)
            total_synergy += synergy
            total_molecules += 1

    avg_synergy = total_synergy / total_molecules if total_molecules > 0 else 0.0
    return {'avg_synergy_score': avg_synergy}

