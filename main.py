import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
import sys
import io
import traceback
from rdkit import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from rdkit import Chem
from rdkit.Chem import  Descriptors
from rdkit import RDLogger
import torch
import argparse
import random
import csv
from datetime import datetime
import json
import logging
import random
from rdkit.Chem import QED
from experiments.experiments import track_performance, assess_polypharmacology_synergy
from utils.save_utils import save_checkpoint, load_checkpoint
from evaluation.evaluator import evaluate_agent, validate
from utils.generate_training_report import generate_training_report
from transform_and_path import ALL_FRAGMENTS, AGING_PATHWAYS, PATHWAY_ACTIVES, MULTI_TARGET_THRESHOLD
from utils.optimizer import optimize_hyperparameters
from agents.CurriculumManager import CurriculumManager, update_curriculum_level
from utils.scorers import create_similarity_scoring_function, predict_docking_score, calculate_sas_score
from utils.pathway_scorers import *
from agents.SpecializedAgent import SpecializedAgent
from agents.MoleculeEnv import MoleculeEnv
from agents.MultiObjectivePrioritizedReplayBuffer import MultiObjectivePrioritizedReplayBuffer
from agents.MultiAgentSystem import MultiAgentSystem
from agents.STaRModel import STaRModel

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')


######################################################################################################################################
###                                         CONFIGURATION                                                                         ###
######################################################################################################################################

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")


######################################################################################################################################
###                                         INITIALIZATION                                                                         ###
######################################################################################################################################



# Initialize similarity scoring functions for all pathways
PATHWAY_SCORING_FUNCTIONS = {
    pathway: create_similarity_scoring_function(pathway)
    for pathway in AGING_PATHWAYS
}



## INITIALIZE PATHWAY SCORING FUNCTIONS
# Aggregating all scoring functions into a dictionary
PATHWAY_SCORING_FUNCTIONS = {
    "Cellular Plasticity Promotion": score_cellular_plasticity_promotion,
    "Proteostasis Enhancement": score_proteostasis_enhancement,
    "DNA Repair Enhancement": score_dna_repair_enhancement,
    "Cellular Reprogramming": score_cellular_reprogramming,
    "Protein Aggregation Inhibition": score_protein_aggregation_inhibition,
    "Genomic Instability Prevention": score_genomic_instability_prevention,
    "Stem Cell Niche Enhancement": score_stem_cell_niche_enhancement,
    "Metabolic Flexibility Enhancement": score_metabolic_flexibility_enhancement,
    "Mitochondrial Dynamics Regulation": score_mitochondrial_dynamics_regulation,
    "Proteolysis Modulation": score_proteolysis_modulation,
    "Telomere Protection": score_telomere_protection,
    "NAD+ Metabolism Modulation": score_nad_metabolism_modulation,
    "Stem Cell Exhaustion Prevention": score_stem_cell_exhaustion_prevention,
    "Autophagy-Lysosomal Pathway Enhancement": score_autophagy_lysosomal_pathway_enhancement,
    "Lipid Metabolism Regulation": score_lipid_metabolism_regulation,
    "Cellular Energy Metabolism Optimization": score_cellular_energy_metabolism_optimization,
    "Cellular Senescence-Associated Secretory Phenotype (SASP) Modulation": score_cellular_senescence_associated_secretory_phenotype_saspm_modulation,
    "Epigenetic Clock Modulation": score_epigenetic_clock_modulation,
}


# Update the group_aging_pathways function to return a dictionary
def group_aging_pathways(num_groups=7):
    pathways = list(PATHWAY_SCORING_FUNCTIONS.keys())
    random.shuffle(pathways)
    groups = [pathways[i::num_groups] for i in range(num_groups)]
    return groups  # Return just the list of groups        
        


######################################################################################################################################
###                                        HELPER FUNCTIONS                                                    ###
######################################################################################################################################

def action_tuple_to_index(atom_index, fragment_index, max_atoms):
    """
    Converts an (atom_index, fragment_index) tuple to a unique integer index.
    """
    return fragment_index * max_atoms + atom_index + 1  # +1 to reserve 0 for 'terminate' action

def index_to_action_tuple(action_index, max_atoms):
    """
    Converts an integer action index back to an (atom_index, fragment_index) tuple.
    """
    fragment_index = (action_index - 1) // max_atoms
    atom_index = (action_index - 1) % max_atoms
    return (atom_index, fragment_index)


def is_complex_molecule(mol):
    # Example criteria for complexity
    mw = Descriptors.ExactMolWt(mol)
    num_rings = Chem.rdMolDescriptors.CalcNumRings(mol)
    return mw > 300 and num_rings > 2



def main():

    parser = argparse.ArgumentParser(description='Molecular RL Agent Training')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file')
    parser.add_argument('--load_optimizer', action='store_true', help='Load optimizer state from checkpoint')
    parser.add_argument('--continue_from_checkpoint', action='store_true', help='Continue training from checkpoint episode')
    parser.add_argument('--total_episodes', type=int, default=200, help='Total number of training episodes')
    parser.add_argument('--curriculum_levels', type=str, default='1,2,2.5,3', help='Comma-separated list of curriculum levels to use')
    parser.add_argument('--seed', type=int, default=543, help='Random seed')
    parser.add_argument('--user_objectives', type=str, required=False, default='{}', help='JSON string of user objectives')
    parser.add_argument('--num_agents', type=int, default=7, help='Number of agents in the multi-agent system')
    parser.add_argument('--alpha', type=float, default=0.1, help='Learning rate for the coordinator')
    args = parser.parse_args()

    # Parse user objectives
    try:
        user_objectives = json.loads(args.user_objectives)
    except json.JSONDecodeError:
        logger.error("Error: Invalid JSON format for user objectives")
        return

    # Parse curriculum levels
    curriculum_levels = [float(level) for level in args.curriculum_levels.split(',')]
    logger.info(f"Using curriculum levels: {curriculum_levels}")

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Optimize hyperparameters
    # Optimize hyperparameters (Removed ChemBERTaAgent)
    objectives = AGING_PATHWAYS  # Replace with actual objectives
    initial_smiles = [           
        'C1=CC=CC=C1',        # Benzene
        'CCO',                # Ethanol
        'CCN',                # Ethylamine
        'CCC',                # Propane
        'CC(=O)O',            # Acetic acid
        'CC(=O)N',            # Acetamide
        'CC#N',               # Acetonitrile
        'C1CCCCC1',           # Cyclohexane
        'C1=CC=CC=C1O',       # Phenol
        'CC(C)O',             # Isopropanol
        "CC1=CC=C(C=C1)NC(=O)C",
        "COC1=C(C=C(C=C1)NC(=O)C)OC",
        "CC(=O)NC1=CC=C(C=C1)O",
        "CC1=C(C(=CC=C1)C)NC(=O)CC2=CC=CC=C2",
        "COC1=C(C=C(C=C1NC(=O)C2=CC=CC=C2)OC)OC",
        "CC1=C(C=CC=C1)NC(=O)CCCN2CCC(CC2)N3C(=O)NC4=CC=CC=C43",
        "CC1=C(C=O)NC1C2=CC=CC=C2",
        "CC1=C(C=CC=C1)S(=O)(=O)NC2=CC=CC=C2",
        "CC1=CC(=NO1)C2=CC(=CC=C2)S(=O)(=O)NC3=CC=CC=C3",
        "CC1=C(C=O)C2=C(C1=O)C(=CC=C2)O",
        "CC1=CC=C(C=C1)OCC(C)(C)NC2=NC=NC3=C2C=CN3",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)CN3CCOCC3)OC",
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)S(=O)(=O)C(C)C)OC",
        "CC(=O)Nc1ccc(cc1)C(=O)O",
        "C1=CC=CC=C1C(=O)O",
        "CCN(CC)C(=O)C1=CC=CC=C1",
        "CC(=O)Nc1ccc(cc1)N",
        "CCOCC(=O)Nc1ccc(cc1)O",
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)S(=O)(=O)C(C)C)OC",
        "CC1=C(C=CC(=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",
        "COC1=C(C=C(C=C1)CNC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)OC"
    ]


    optimized_params = optimize_hyperparameters(
        agent_class=SpecializedAgent, 
        env_class=MoleculeEnv, 
        objectives=objectives,
        initial_smiles=initial_smiles,
        n_iter=50
    )
    logger.info(f"Optimized hyperparameters: {optimized_params}")

    # Define known initially active molecules
    known_molecules = [
        'C1=CC=CC=C1',        # Benzene
        'CNC(=O)C1=CC=CC=C1', # Benzamide
        'CC(=O)O',            # Acetic acid
        'C1CCCCC1',           # Cyclohexane
        'C1=CC=C(C=C1)O',     # Phenol
        'CC(C)O',             # Isopropanol
        'CCCC',               # Butane
        'C1=CN=CN=C1',        # Pyrimidine
        'C1=CC=NC=C1',        # Pyridine
        'C1CCNCC1',           # Piperidine
        'C1COCCO1',           # 1,4-Dioxane
        'CC#N',               # Acetonitrile
        'CCN',                # Ethylamine
        'CCOCC',              # Diethyl ether
        # Add more known molecule SMILES as needed
    ]

    # Create a unique output directory for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"molecular_rl_agent_model_{timestamp}"
    output_dir = os.path.join("logs", "training_reports", model_name)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")

    # Define paths within the output directory
    log_file = os.path.join(output_dir, f"{model_name}_training_log.csv")
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    report_dir = os.path.join(output_dir, "reports")
    os.makedirs(report_dir, exist_ok=True)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Model directory: {model_dir}")
    logger.info(f"Report directory: {report_dir}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    checkpoint_interval = 100  # Save checkpoint every 100 episodes
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"Checkpoint directory: {checkpoint_dir}")

    # Group aging pathways
    pathway_groups = group_aging_pathways(num_groups=args.num_agents)
    logger.info(f"Pathway groups: {pathway_groups}")

    curriculum_manager = CurriculumManager(
        complexity_threshold=300, 
        reward_threshold=10, 
        complex_molecule_count=5,
        window_size=5,  # Adjust based on desired sensitivity
        num_levels=args.num_agents  # Assuming one level per agent
    )
    logger.info(f"Initialized CurriculumManager with {args.num_agents} levels")

    # Initialize MultiAgentSystem
    multi_agent_system = MultiAgentSystem(
        num_agents=args.num_agents,
        env=None,  # Environments are initialized within the MultiAgentSystem
        action_size=100,  # Adjust based on actual action space
        learning_rate=optimized_params['learning_rate'],
        discount_factor=optimized_params['discount_factor'],
        epsilon_decay=optimized_params['epsilon_decay'],
        objectives=objectives,
        pathway_groups=pathway_groups,
        user_objectives=user_objectives,
        known_molecules_smiles=known_molecules,
        alpha=args.alpha,
        batch_size=optimized_params['batch_size']
    )
    logger.info(f"Initialized MultiAgentSystem with {args.num_agents} agents")
    logger.info(f"Learning rate: {optimized_params['learning_rate']}")
    logger.info(f"Discount factor: {optimized_params['discount_factor']}")
    logger.info(f"Epsilon decay: {optimized_params['epsilon_decay']}")
    logger.info(f"Batch size: {optimized_params['batch_size']}")
    
    
    # Initialize schedulers for each agent
    schedulers = []
    for agent in multi_agent_system.agents:
        scheduler = ReduceLROnPlateau(agent.optimizer, mode='max', factor=0.5, patience=10, verbose=True)
        schedulers.append(scheduler)
    logger.info(f"Initialized {len(schedulers)} schedulers")

    # Shared MultiObjectivePrioritizedReplayBuffer
    replay_buffer = MultiObjectivePrioritizedReplayBuffer(capacity=100000)
    logger.info(f"Initialized replay buffer with capacity: 100000")

    # Training Loop Initialization
    start_episode = 0
    total_reward = 0
    running_reward = 0
    log_data = []
    fieldnames = ['Episode', 'Agent_ID', 'Curriculum_Level', 'Reward', 'Running_Reward', 'SMILES_Sequence', 'Objectives', 'Pathway_Score', 'Diversity_Score', 'Multitarget_Score', 'Rationale', 'Avg_Complexity', 'QED_Score', 'Lipinski_Violations', 'SA_Score', 'Synergy_Score', 'Individual_Objectives']
    # Write CSV header
    with io.open(log_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    logger.info(f"Initialized CSV log file with headers")

    batch_size = 64
    accumulation_steps = 4  # Accumulate gradients over 4 batches
    star_model = STaRModel()
    logger.info(f"Initialized STaRModel")


    max_steps_per_episode = 1000
    max_steps = 1000
    for episode in range(start_episode, args.total_episodes):
        try:
            # Reset environments and initialize tracking variables
            states = [agent.env.reset() for agent in multi_agent_system.agents]
            dones = [False] * args.num_agents
            ep_rewards = [0] * args.num_agents
            complex_molecules = [0] * args.num_agents
            episode_total_reward = 0  # Track total reward for this episode
            step = 0

            while not all(dones) and step < max_steps:
                step += 1
                actions = []
                for idx, agent in enumerate(multi_agent_system.agents):
                    if not dones[idx]:
                        action = agent.select_action(states[idx])
                        actions.append(action)
                    else:
                        actions.append(None)
                logger.debug(f"Step {step}: Actions selected: {actions}")


                next_states = []
                experiences = []
                for idx, agent in enumerate(multi_agent_system.agents):
                    if not dones[idx]:
                        next_state, reward, done, info = agent.env.step(actions[idx])
                        experiences.append((states[idx], actions[idx], reward, next_state, done, idx))
                        replay_buffer.push((states[idx], actions[idx], reward, next_state, done, idx))
                        states[idx] = next_state
                        ep_rewards[idx] += reward
                        dones[idx] = done
                        episode_total_reward += reward

                        # Track complexity based on molecular weight
                        mol = agent.env.current_mol
                        if mol and Descriptors.ExactMolWt(mol) >= curriculum_manager.complexity_threshold:
                            complex_molecules[idx] += 1
                        logger.debug(f"Agent {idx}: Reward: {reward}, Done: {done}, Molecule Weight: {Descriptors.ExactMolWt(mol) if mol else 'N/A'}")
                    else:
                        next_state = states[idx]
                    next_states.append(next_state)

                # Update agents with experiences
                if len(replay_buffer) >= batch_size:
                    sampled_experiences, weights, indices = replay_buffer.sample(batch_size)
                    if len(sampled_experiences) < batch_size:
                        logger.debug(f"Skipping update: Not enough samples. Got {len(sampled_experiences)}, need {batch_size}")
                        continue  # Skip if not enough samples
                    multi_agent_system.update(sampled_experiences, weights, indices, replay_buffer)
                    logger.debug(f"Updated agents with batch size: {len(sampled_experiences)}")

            # End of episode processing
            running_reward = 0.05 * episode_total_reward + (1 - 0.05) * running_reward
            total_reward += episode_total_reward

            logger.info(f"Finished episode {episode} after {step} steps")
            avg_complexity = [c / step for c in complex_molecules]
            logger.info(f"Episode {episode}: Rewards = {ep_rewards}, Avg Complexity = {avg_complexity}")

            # Update the curriculum based on rewards and complexity
            curriculum_manager.update_curriculum(episode, ep_rewards, complex_molecules)

            # Get the current curriculum level
            current_level = curriculum_manager.get_current_level()
            logger.info(f"Current Curriculum Level: {current_level}")

            # Set curriculum level for each agent's environment
            for agent in multi_agent_system.agents:
                agent.env.set_curriculum_level(current_level)
                
            # Generate rationales and log data
            for idx, agent in enumerate(multi_agent_system.agents):
                final_state = agent.env.get_state()
                mol_repr = agent.get_molecule_representation(final_state)
                prompt = f"Given the molecular state: {final_state}, provide a brief rationale for the agent's actions in generating this molecule. Focus on its potential effects on aging pathways and its drug-like properties."
                rationale = star_model.generate_rationale(f"State: {final_state}", mol_repr.detach(), agent.env, agent.objectives)
                logger.debug(f"After generate_rationale: Agent {idx}, rationale: {rationale[:50]}...")
                star_model.add_to_buffer(f"State: {final_state}", rationale, mol_repr.detach())

                # Update the scheduler
                scheduler = schedulers[idx]
                scheduler.step(ep_rewards[idx])

                # Calculate pathway and multitarget scores
                try:
                    # Calculate pathway and multitarget scores
                    mol = agent.env.current_mol
                    pathway_scores = agent.env._get_pathway_scores(mol)
                    
                    # Calculate individual objective scores
                    individual_objectives = {obj: score for obj, score in zip(agent.objectives, pathway_scores) if score is not None}
                    
                    # Calculate additional metrics
                    qed_score = QED.qed(mol) if mol else 0.0
                    lipinski_violations = agent.env._lipinski_violations(mol) if mol else 0
                    sa_score = calculate_sas_score(mol) if mol else 0.0
                    synergy_score = assess_polypharmacology_synergy([mol], agent.objectives) if mol else 0.0
                    
                    pathway_score = np.mean([score for score in pathway_scores if score is not None]) if pathway_scores else 0.0

                    logger.debug(f"Agent {idx}: QED Score: {qed_score}, Lipinski Violations: {lipinski_violations}, SA Score: {sa_score}, Synergy Score: {synergy_score}")
                    logger.debug(f"Agent {idx}: Pathway Score: {pathway_score}, Individual Objectives: {individual_objectives}")

                except Exception as e:
                    logger.error(f"Error calculating metrics for Agent {idx} in episode {episode}: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    pathway_score = 0.0
                    individual_objectives = {obj: 0.0 for obj in agent.objectives}
                    qed_score = 0.0
                    lipinski_violations = 0
                    sa_score = 0.0
                    synergy_score = 0.0
                    
    
                # Prepare log entry
                log_entry = {
                    'Episode': episode,
                    'Agent_ID': idx,
                    'Curriculum_Level': agent.env.curriculum_level,
                    'Reward': ep_rewards[idx],
                    'Running_Reward': running_reward,
                    'SMILES_Sequence': '|'.join(agent.env.smiles_sequence),
                    'Objectives': ', '.join(agent.objectives),
                    'Pathway_Score': pathway_score,
                    'Diversity_Score': agent.env._calculate_diversity(),
                    'Rationale': rationale if rationale is not None else "No rationale generated",
                    'Avg_Complexity': avg_complexity[idx],
                    'QED_Score': qed_score,
                    'Diversity_Score': agent.env.diversity_score,
                    'Multitarget_Score': agent.env.multitarget_score,
                    'Individual_Objectives': json.dumps(individual_objectives),
                    'Lipinski_Violations': lipinski_violations,
                    'SA_Score': sa_score,
                    'Synergy_Score': synergy_score,
                    'Individual_Objectives': json.dumps(individual_objectives),

                }
                log_data.append(log_entry)
                logger.debug(f"Prepared log entry for Agent {idx}: {log_entry}")

            # Logging to CSV
            if episode % 1 == 0:
                try:
                    with open(log_file, 'a', newline='', encoding='utf-8') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writerows(log_data)
                    log_data = []  # Clear log data after writing
                except Exception as e:
                    logger.error(f"Error writing to CSV in episode {episode}: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")

            # Print progress every episode
            if episode % 1 == 0 or episode == start_episode + 1:
                for idx, agent in enumerate(multi_agent_system.agents):
                    logger.info(f"Episode {episode}, Agent {idx} completed with reward {ep_rewards[idx]:.4f}.")
                    logger.info(f"Running Reward: {running_reward:.4f}")
                    logger.info(f"Curriculum Level: {agent.env.curriculum_level}, SMILES: {log_entry['SMILES_Sequence']}")
                    logger.info(f"Objectives: {log_entry['Objectives']}")
                    logger.info(f"Rationale: {rationale}")          
                    print(f"Completed episode {episode}")

            logger.info(f"Finished episode {episode}")
            if episode % 100 == 0:  # Clear cache every 100 episodes
                torch.cuda.empty_cache()
                
            # Perform Periodic Validations
            if episode % 50 == 0:
                logger.info(f"Performing validation for episode {episode}")
                validation_results = {}
                for agent in multi_agent_system.agents:
                    results = validate(agent, agent.env)
                    validation_results[agent.objectives] = results
                    logger.info(f"Validation results for Agent {agent.objectives} at episode {episode}:")
                    logger.info(results)


            # Save Checkpoints
            if episode % 100 == 0:
                logger.info(f"Saving checkpoint for episode {episode}")
                for idx, agent in enumerate(multi_agent_system.agents):
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_episode_{episode}_agent_{idx}.pth.tar")
                    save_checkpoint({
                        'episode': episode,
                        'state_dict': agent.model.state_dict(),
                        'optimizer_state_dict': agent.optimizer.state_dict(),
                    }, filename=checkpoint_path)

        except Exception as e:
            logger.error(f"An error occurred in episode {episode}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            if episode % 10 == 0:  # Save checkpoint every 10 episodes in case of error
                for idx, agent in enumerate(multi_agent_system.agents):
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_episode_{episode}_agent_{idx}_error.pth.tar")
                    save_checkpoint({
                        'episode': episode,
                        'state_dict': agent.model.state_dict(),
                        'optimizer_state_dict': agent.optimizer.state_dict(),
                    }, filename=checkpoint_path)
            continue  # Continue to the next episode instead of breaking the loop


    logger.info("Training loop completed. Starting final processes.")

    try:

        logger.info("Final training of STaR model on collected data...")
        star_model.train_on_buffer()
        logger.info(f"Number of log entries: {len(log_data)}")
        logger.info("Generating training report...")

        logger.info(f"Number of log entries: {len(log_data)}")
        if len(log_data) > 0:
            logger.info("Generating training report...")
            generate_training_report(log_file, log_data, output_dir=output_dir, best_run_path='best_run.json')
        else:
            logger.warning("No log data available. Skipping training report generation.")
        logger.info("Saving final models...")
        for idx, agent in enumerate(multi_agent_system.agents):
            try:
                agent_name = "_".join(agent.objectives).replace(" ", "_")[:50]
                final_checkpoint = os.path.join(model_dir, f"{model_name}_final_model_agent_{agent_name}.pth.tar")
                save_checkpoint({
                    'episode': args.total_episodes, 
                    'state_dict': agent.model.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                }, filename=final_checkpoint)
                logger.info(f"Saved checkpoint for Agent: {agent_name}")
            except Exception as e:
                logger.error(f"Error saving checkpoint for Agent {idx}: {str(e)}")

        logger.info(f"Training completed. Total reward: {total_reward:.4f}")
        logger.info(f"Average reward: {total_reward / (args.total_episodes * args.num_agents):.4f}")

    except Exception as e:
        logger.error(f"An error occurred during final processes: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    logger.info("Program finished executing.")
    
    
if __name__ == "__main__":
    main()
