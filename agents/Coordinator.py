import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Coordinator:
    def __init__(self, num_agents, alpha=0.1):
        self.num_agents = num_agents
        self.weights = nn.Parameter(torch.ones(num_agents) / num_agents)
        self.alpha = alpha
        self.optimizer = torch.optim.Adam([self.weights], lr=alpha)
        self.q_values = np.zeros(num_agents)

    def combine_actions(self, actions):
        softmax_weights = F.softmax(self.weights, dim=0)
        weighted_actions = [w * a for w, a in zip(softmax_weights, actions)]
        return sum(weighted_actions)

    def update(self, experiences, weights=None, indices=None, replay_buffer=None):
        # Aggregate rewards with explicit floating-point dtype
        rewards = torch.tensor([exp[2] for exp in experiences], dtype=torch.float32, device=self.weights.device)
        
        # Debug: Check dtype
        print(f"Debug: rewards tensor dtype: {rewards.dtype}")

        # Normalize rewards
        normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Calculate the mean reward for each agent
        agent_rewards = torch.zeros(self.num_agents, device=self.weights.device)
        for i, exp in enumerate(experiences):
            agent_id = exp[-1] if len(exp) > 5 else 0  # Use last element as agent ID if available
            agent_rewards[agent_id] += normalized_rewards[i]
        agent_rewards /= len(experiences) / self.num_agents  # Average reward per agent
        
        # Update weights
        self.optimizer.zero_grad()
        loss = -torch.sum(F.softmax(self.weights, dim=0) * agent_rewards)
        loss.backward()
        self.optimizer.step()
        
        # Re-normalize weights using softmax to ensure they sum to 1
        with torch.no_grad():
            self.weights.data = F.softmax(self.weights.data, dim=0)
        
        # Update Q-values
        for exp in experiences:
            agent_id = exp[-1] if len(exp) > 5 else 0
            reward = exp[2]
            self.q_values[agent_id] += self.alpha * (reward - self.q_values[agent_id])
    
    def select_agent(self):
        return np.argmax(self.q_values)