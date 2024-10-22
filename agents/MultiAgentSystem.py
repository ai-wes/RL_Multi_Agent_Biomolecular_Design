
from agents.Coordinator import Coordinator
from agents.SpecializedAgent import SpecializedAgent
from agents.MoleculeEnv import MoleculeEnv
from transform_and_path import PATHWAY_SCORING_FUNCTIONS



class MultiAgentSystem:
    def __init__(self, num_agents, env, action_size, learning_rate, discount_factor, epsilon_decay, objectives, pathway_groups, user_objectives, known_molecules_smiles, alpha, batch_size):
        self.action_space_size = 100  # Set this to the correct value based on your environment
        self.agents = []
        self.envs = []
        
        for pathway_group in pathway_groups:
            env = MoleculeEnv(
                max_steps=20,
                max_atoms=30,
                curriculum_level=1,
                pathway_scoring_functions={pathway: PATHWAY_SCORING_FUNCTIONS[pathway] for pathway in pathway_group},
                user_objectives=user_objectives,
                agent_objectives=pathway_group,
                known_molecules_smiles=known_molecules_smiles
            )
            self.envs.append(env)

            agent = SpecializedAgent(
                action_size=action_size,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon_decay=epsilon_decay,
                env=env,
                objectives=pathway_group
            )
            self.agents.append(agent)
        
        self.coordinator = Coordinator(len(self.agents), alpha)
        self.batch_size = batch_size


    def update(self, experiences, weights=None, indices=None, replay_buffer=None):
        for agent_id, agent in enumerate(self.agents):
            agent_experiences = [exp for exp in experiences if exp[-1] == agent_id]
            if agent_experiences:
                agent.update_batch(agent_experiences, weights, indices, replay_buffer)
        
        # Update the coordinator with all experiences
        self.coordinator.update(experiences, weights, indices, replay_buffer)

    def select_action(self, states):
        actions = [agent.select_action(state) for agent, state in zip(self.agents, states)]
        return self.coordinator.combine_actions(actions)
    
    
    