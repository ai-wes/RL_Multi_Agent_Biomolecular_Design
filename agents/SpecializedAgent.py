


from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from agents.ChemBERTaAgent import ChemBERTaAgent
from transform_and_path import AGING_PATHWAYS, MULTI_TARGET_THRESHOLD
from utils.scorers import predict_docking_score


class SpecializedAgent(ChemBERTaAgent):
    def __init__(self, action_size, learning_rate, discount_factor, epsilon_decay, env, objectives):
        super().__init__(action_size, learning_rate, discount_factor, epsilon_decay, env, objectives)
   
    def compute_reward(self, env, state, action, next_state):
        rewards = []
        
        next_mol = self.state_to_mol(next_state)
        
        for objective in self.objectives:
            if objective == 'docking':
                docking_score = -predict_docking_score(Chem.MolToSmiles(next_mol))
                rewards.append(docking_score)
                logger.debug(f"Docking score: {docking_score}")
            else:  # pathway objective
                pathway_scores = env._get_pathway_scores(next_mol)
                objective_index = AGING_PATHWAYS.index(objective)
                pathway_score = pathway_scores[objective_index]
                rewards.append(pathway_score)
                logger.debug(f"Pathway score for {objective}: {pathway_score}")
        
        prev_mol = self.state_to_mol(state)
        prev_reward = self.calculate_objective_reward(prev_mol)
        next_reward = self.calculate_objective_reward(next_mol)
        
        reward = (next_reward - prev_reward) - 0.01  # Small penalty for each action
        
        # Add complexity bonus from the environment
        complexity_bonus = env.calculate_complexity_bonus(action)
        reward += complexity_bonus
        
        logger.debug(f"Reward breakdown: prev_reward={prev_reward}, next_reward={next_reward}, complexity_bonus={complexity_bonus}, final_reward={reward}")
        
        return reward
    
    
    def calculate_objective_reward(self, mol):
        rewards = []
        for objective in self.objectives:
            if objective == 'docking':
                rewards.append(-predict_docking_score(Chem.MolToSmiles(mol)))
            else:  # pathway objective
                pathway_scores = self.env._get_pathway_scores(mol)
                objective_index = AGING_PATHWAYS.index(objective)
                # Only add the pathway score if it exceeds the threshold
                if pathway_scores[objective_index] > MULTI_TARGET_THRESHOLD:
                    rewards.append(pathway_scores[objective_index])
                else:
                    rewards.append(0.0)
        return np.mean(rewards) if rewards else 0.0

    def state_to_mol(self, state):
        # Convert state (fingerprint) to a unique string representation
        state_key = ''.join(map(str, state))
        
        # Check if we've already converted this state
        if state_key in self.state_mol_cache:
            return self.state_mol_cache[state_key]
        
        # If not, we need to reconstruct the molecule
        # This assumes that the state is a Morgan fingerprint
        # and that we're using ECFP4 (radius 2)
        all_mols = self.env.get_all_mols()  # Ensure this method is correctly implemented
        
        best_match = None
        best_similarity = -1
        
        for mol in all_mols:
            mol_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=len(state))
            similarity = DataStructs.TanimotoSimilarity(mol_fp, state)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = mol
        
        # Cache the result
        self.state_mol_cache[state_key] = best_match
        
        return best_match

    def update_state_mol_cache(self, state, mol):
        # Call this method whenever a new state-molecule pair is created
        state_key = ''.join(map(str, state))
        self.state_mol_cache[state_key] = mol
