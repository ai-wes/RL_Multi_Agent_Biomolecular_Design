import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from evaluation.evaluator import evaluate_agent





class ChemBERTaAgent:
    def __init__(self, action_size, learning_rate, discount_factor, epsilon_decay, env, objectives=None):
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = discount_factor
        self.epsilon_decay = epsilon_decay
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.env = env
        self.objectives = objectives if objectives is not None else []
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Load pre-trained ChemBERTa model
        self.tokenizer = AutoTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
        self.model = AutoModelForMaskedLM.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
        # Move the model to the specified device
        self.model = self.model.to(self.device)
        # Assuming we have maximum limits for atom_index and fragment_index
        self.max_atom_index = env.max_atoms  # Define appropriately
        self.max_fragment_index = len(env.all_fragments)  # Define appropriately

        # Create mappings
        self.action_to_index = {}
        self.index_to_action = {}
        index = 0
        for atom_index in range(self.max_atom_index):
            for frag_idx in range(len(env.all_fragments)):
                self.action_to_index[(atom_index, frag_idx)] = index
                self.index_to_action[index] = (atom_index, frag_idx)
                index += 1
        self.action_size = index  # Update action_size based on the total number of unique actions

        # Initialize Q-head with the updated action_size
        self.q_head = nn.Linear(2048, self.action_size).to(self.device)
        # Re-initialize optimizer if necessary
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
                
        # Ensure epsilon is set correctly
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay_rate = self.epsilon_decay

    def get_valid_actions(self, state):
        valid_actions = []
        for action in range(self.action_size):
            # Reverse mapping to get (atom_index, frag_idx)
            atom_index, frag_idx = self.index_to_action[action]
            
            # Add your validation logic here
            if atom_index < self.max_atom_index and frag_idx < len(self.env.all_fragments):
                valid_actions.append(action)  # Append the unique index
        return valid_actions


    def get_molecule_representation(self, state):
        with torch.no_grad():
            if isinstance(state, list):
                # If state is a list, process each element
                return torch.stack([self._process_single_state(s) for s in state])
            else:
                # If state is a single item, process it directly
                return self._process_single_state(state)



    def _process_single_state(self, state):
        if isinstance(state, np.ndarray):
            return torch.from_numpy(state).float().to(self.device)
        elif isinstance(state, torch.Tensor):
            return state.to(self.device)
        elif isinstance(state, str):
            inputs = self.tokenizer(state, return_tensors='pt', padding=True, truncation=True).to(self.device)
            outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).squeeze(0)
        else:
            raise ValueError(f"Unexpected state type: {type(state)}")



    def select_action(self, state):
        valid_actions = self.env.get_valid_actions(state)
        if not valid_actions:
            print("Warning: No valid actions available. Returning default action.")
            return 0  # Return a default action (you might want to adjust this based on your action space)

        if np.random.rand() <= self.epsilon:
            # Epsilon-greedy exploration
            valid_action_indices = [self.action_to_index[action] for action in valid_actions]            
            # Bias towards more complex fragments
            action_weights = [self.env.fragment_complexity[self.env.all_fragments[action[1]]] for action in valid_actions]
            action_weights = np.array(action_weights)
            action_weights = action_weights / np.sum(action_weights)
            
            chosen_index = np.random.choice(len(valid_action_indices), p=action_weights)
            action_index = valid_action_indices[chosen_index]
            return action_index
        else:
            with torch.no_grad():
                mol_repr = self.get_molecule_representation(state)
                if mol_repr.dim() == 1:
                    mol_repr = mol_repr.unsqueeze(0)  # Add batch dimension if it's a single state
                q_values = self.q_head(mol_repr)  # [batch_size, action_size]
                return int(torch.argmax(q_values).item())  # Always return a single action index            
            
                
                
    def update_batch(self, experiences, weights=None, indices=None, replay_buffer=None):
        if len(experiences[0]) == 5:  # If agent_ids are not present
            states, actions, rewards, next_states, dones = zip(*experiences)
            agent_ids = [0] * len(experiences)  # Use 0 as a placeholder
        else:
            states, actions, rewards, next_states, dones, agent_ids = zip(*experiences)
        
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)

        # Map action tuples to indices, handling invalid actions
        action_indices = []
        for action in actions:
            if action in self.action_to_index:
                action_indices.append(self.action_to_index[action])
            else:
                action_indices.append(0)  # Use 0 as a default index for invalid actions
        actions = torch.tensor(action_indices, dtype=torch.long).to(self.device)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        if weights is not None:
            weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        else:
            weights = torch.ones_like(rewards)

        q_values = self.q_head(states)  # [batch_size, action_size]
        actions = actions.view(-1, 1)  # Ensure actions have the correct shape
        current_q_values = q_values.gather(1, actions).squeeze(1)  # [batch_size]

        # Compute target Q-values
        next_q_values = self.q_head(next_states).max(1)[0]  # [batch_size]
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Check for shape mismatch
        if current_q_values.shape != target_q_values.shape:
            raise ValueError("Shape mismatch between current_q_values and target_q_values")

        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values.detach(), reduction='none')
        loss = (loss * weights).mean()
        print(f"Debug: loss: {loss.item()}")

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Compute individual losses for priority update
        individual_losses = F.mse_loss(current_q_values, target_q_values.detach(), reduction='none')

        # Update priorities in the replay buffer if provided
        if replay_buffer is not None and indices is not None:
            priorities = (individual_losses.detach().cpu().numpy() + 1e-6).tolist()
            replay_buffer.update_priorities(indices, priorities)

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        print(f"Debug: epsilon: {self.epsilon}")

        return loss.item()