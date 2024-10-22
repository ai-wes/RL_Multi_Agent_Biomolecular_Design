


from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from transform_and_path import MULTI_TARGET_THRESHOLD
from rdkit import Chem

def is_valid_molecule(mol):
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol)
        return smiles != ''
    except:
        return False



def validate(agent, env, n_episodes=10):
    total_reward = 0
    valid_molecules = 0
    diversity_score = 0
    pathway_scores = []
    multitarget_scores = []

    for _ in range(n_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state

        total_reward += episode_reward

        # Check molecule validity
        mol = env.current_mol
        if Chem.MolToSmiles(mol) != '':
            valid_molecules += 1

        # Calculate diversity
        diversity_score += env._calculate_diversity()

        # Calculate pathway scores
        current_pathway_scores = env._get_pathway_scores(mol)
        pathway_scores.append(current_pathway_scores)

        # Calculate multitarget score using standardized threshold
        if agent.objectives:
            multitarget_score = sum(score > MULTI_TARGET_THRESHOLD for score in current_pathway_scores) / len(agent.objectives)
        else:
            multitarget_score = 0.0
        multitarget_scores.append(multitarget_score)

    avg_reward = total_reward / n_episodes
    validity_rate = valid_molecules / n_episodes
    avg_diversity = diversity_score / n_episodes
    avg_pathway_scores = np.mean(pathway_scores, axis=0) if pathway_scores else []
    avg_multitarget_score = np.mean(multitarget_scores) if multitarget_scores else 0.0

    return {
        'avg_reward': avg_reward,
        'validity_rate': validity_rate,
        'avg_diversity': avg_diversity,
        'avg_pathway_scores': avg_pathway_scores,
        'avg_multitarget_score': avg_multitarget_score
    }




# evaluation/evaluator.py
import logging

# evaluation/evaluator.py
import logging

def evaluate_agent(agent, env, batch_size):
    logger = logging.getLogger(__name__)
    total_reward = 0.0
    num_episodes = 20
    training_episodes = 100
    
    experiences = []  # List to accumulate experiences
    
    for episode in range(num_episodes):
        # Training loop
        for train_ep in range(training_episodes):
            state = env.reset()
            done = False
            train_reward = 0.0
            episode_experiences = []  # Experiences for this episode
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                episode_experiences.append((state, action, reward, next_state, done))
                state = next_state
                train_reward += reward
            
            experiences.extend(episode_experiences)
            
            # If we have enough experiences, perform a batch update
            if len(experiences) >= batch_size:
                agent.update_batch(experiences[:batch_size])
                experiences = experiences[batch_size:]  # Remove used experiences
            
            logger.debug(f"Training Episode {train_ep+1}/{training_episodes}: Reward={train_reward}")
        
        # Evaluation loop
        state = env.reset()
        episode_reward = 0.0
        done = False
        step = 0
        while not done:
            action = agent.select_action(state)  # Removed 'evaluate=True'
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
            step += 1
            logger.debug(f"Evaluation Episode {episode+1}, Step {step}: Action={action}, Reward={reward}")
        
        logger.info(f"Evaluation Episode {episode+1}: Total Reward = {episode_reward}")
        total_reward += episode_reward
    
    # Perform final update with remaining experiences
    if experiences:
        agent.update_batch(experiences)
    
    average_reward = total_reward / num_episodes
    logger.info(f"Average Reward over {num_episodes} evaluation episodes: {average_reward}")
    
    return average_reward