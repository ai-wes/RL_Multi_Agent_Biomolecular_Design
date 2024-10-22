import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



import argparse
import logging
import json
from ChemBERTaAgent import ChemBERTaAgent
from MoleculeEnv import MoleculeEnv
from evaluation.evaluator import evaluate_agent
from transform_and_path import PATHWAY_SCORING_FUNCTIONS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(args):
    logger.info(f"Agent objectives: {args.objectives}")
    logger.info(f"Initial SMILES: {args.initial_smiles}")
    
    env = MoleculeEnv(
        max_steps=args.max_steps,
        max_atoms=args.max_atoms,
        curriculum_level=args.curriculum_level,
        pathway_scoring_functions={pathway: PATHWAY_SCORING_FUNCTIONS[pathway] for pathway in args.objectives},
        user_objectives=args.user_objectives,
        agent_objectives=args.objectives,
        known_molecules_smiles=args.initial_smiles
    )
    
    
    agent = ChemBERTaAgent(
        action_size=env.action_space_size,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        epsilon_decay=args.epsilon_decay,
        env=env,
        objectives=args.objectives
    )
    
    average_reward = evaluate_agent(agent, env, batch_size=args.batch_size)
    logger.info(f"Average Reward: {average_reward}")
    
    result = {
        'average_reward': average_reward,
        'hyperparameters': {
            'learning_rate': args.learning_rate,
            'discount_factor': args.discount_factor,
            'epsilon_decay': args.epsilon_decay,
            'batch_size': args.batch_size
        }
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and evaluate a single agent.")
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--discount_factor', type=float, default=0.95)
    parser.add_argument('--epsilon_decay', type=float, default=0.99)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_steps', type=int, default=100)
    parser.add_argument('--max_atoms', type=int, default=30)
    parser.add_argument('--curriculum_level', type=int, default=1)
    parser.add_argument('--objectives', nargs='+', required=True)
    parser.add_argument('--user_objectives', nargs='*', default=[])
    parser.add_argument('--initial_smiles', nargs='+', required=True)
    parser.add_argument('--train_episodes', type=int, default=100)
    parser.add_argument('--output_file', type=str, default='single_agent_result.json')
    
    args = parser.parse_args()
    main(args)