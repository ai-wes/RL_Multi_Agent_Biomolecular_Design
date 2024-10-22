import logging
from evaluation.evaluator import evaluate_agent
from transform_and_path import PATHWAY_SCORING_FUNCTIONS
from skopt import gp_minimize
from skopt.space import Real, Integer

import logging
from evaluation.evaluator import evaluate_agent
from transform_and_path import PATHWAY_SCORING_FUNCTIONS
from skopt import gp_minimize
from skopt.space import Real, Integer

logger = logging.getLogger(__name__)

def optimize_hyperparameters(agent_class, env_class, objectives, initial_smiles, n_iter=50):
    def objective(params):
        learning_rate, discount_factor, epsilon_decay, batch_size = params
        
        logger.info(f"Evaluating with Params: LR={learning_rate:.2e}, DF={discount_factor:.4f}, ED={epsilon_decay:.4f}, BS={batch_size}")
        
        env = env_class(
            max_steps=100,
            max_atoms=30,
            curriculum_level=1,
            pathway_scoring_functions={pathway: PATHWAY_SCORING_FUNCTIONS[pathway] for pathway in objectives},
            user_objectives={},
            agent_objectives=objectives,
            known_molecules_smiles=initial_smiles
        )
        
        agent = agent_class(
            action_size=env.action_space_size,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            epsilon_decay=epsilon_decay,
            env=env,
            objectives=objectives
        )
        
        average_reward = evaluate_agent(agent, env, batch_size=batch_size)
        logger.info(f"Average Reward: {average_reward}")
        
        return -average_reward

    param_space = [
        Real(1e-6, 1e-2, name='learning_rate', prior='log-uniform'),
        Real(0.7, 0.9999, name='discount_factor'),
        Real(0.8, 0.9999, name='epsilon_decay'),
        Integer(16, 512, name='batch_size')
    ]

    result = gp_minimize(objective, param_space, n_calls=100, random_state=42, verbose=True)
    
    best_params = {
        'learning_rate': result.x[0],
        'discount_factor': result.x[1],
        'epsilon_decay': result.x[2],
        'batch_size': result.x[3]
    }
    
    logger.info(f"Best parameters found: {best_params}")
    logger.info(f"Best reward: {-result.fun}")
    
    return best_params