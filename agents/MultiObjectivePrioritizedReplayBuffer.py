import numpy as np

class MultiObjectivePrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.prioritized = True

    def push(self, experience):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        self.frame += 1

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        if buffer_size < batch_size:
            indices = np.arange(buffer_size)
            weights = np.ones(buffer_size)
            return self.buffer, weights, indices
        
        priorities = self.priorities[:buffer_size] ** self.alpha
        probs = priorities / priorities.sum()
        
        indices = np.random.choice(buffer_size, batch_size, p=probs, replace=False)
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        weights = (buffer_size * probs[indices]) ** -beta
        weights /= weights.max()
        
        sampled_experiences = [self.buffer[i] for i in indices]
        return sampled_experiences, weights, indices
    
    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio
    
    def __len__(self):
        return len(self.buffer)
