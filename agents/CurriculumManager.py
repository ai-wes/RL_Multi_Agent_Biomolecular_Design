



import logging
import numpy as np

logger = logging.getLogger(__name__)


# In the update_curriculum_level function
def update_curriculum_level(episode, episodes_per_level, levels):
    current_level_index = min(episode // episodes_per_level, len(levels) - 1)
    return levels[current_level_index]





class CurriculumManager:
    def __init__(self, complexity_threshold, reward_threshold, complex_molecule_count, window_size=5, num_levels=4):
        """
        Initializes the CurriculumManager.

        Args:
            complexity_threshold (float): Minimum molecular weight to consider a molecule complex.
            reward_threshold (float): Minimum average reward to consider advancing the curriculum.
            complex_molecule_count (int): Minimum number of complex molecules to consider advancing.
            window_size (int): Number of recent episodes to consider for moving averages.
            num_levels (int): Total number of curriculum levels.
        """
        self.current_level = 1
        self.complexity_threshold = complexity_threshold
        self.reward_threshold = reward_threshold
        self.complex_molecule_count = complex_molecule_count
        self.window_size = window_size
        self.levels = [1, 2, 2.5, 3]  # Example levels; adjust as needed
        self.num_levels = num_levels

        # Initialize buffers to store recent rewards and complex molecule counts
        self.recent_rewards = []
        self.recent_complex_counts = []

    def update_curriculum(self, episode, rewards, complex_molecules):
        """
        Updates the curriculum level based on recent rewards and complex molecule counts.

        Args:
            episode (int): Current episode number.
            rewards (list of float): Rewards obtained by agents in the current episode.
            complex_molecules (list of int): Number of complex molecules produced by agents in the current episode.
        """
        # Aggregate rewards and complex molecule counts
        avg_reward = np.mean(rewards)
        total_complex = np.sum(complex_molecules)

        # Update recent history
        self.recent_rewards.append(avg_reward)
        self.recent_complex_counts.append(total_complex)

        if len(self.recent_rewards) > self.window_size:
            self.recent_rewards.pop(0)
            self.recent_complex_counts.pop(0)

        # Calculate moving averages
        moving_avg_reward = np.mean(self.recent_rewards)
        moving_avg_complex = np.mean(self.recent_complex_counts)  # Using mean to smooth variability

        logger.info(f"Episode {episode}: Moving Avg Reward = {moving_avg_reward:.2f}, Moving Avg Complex Molecules = {moving_avg_complex:.2f}")

        # Check if thresholds are met
        if (moving_avg_reward >= self.reward_threshold) and (moving_avg_complex >= self.complex_molecule_count):
            # Check if not already at max level
            if self.current_level < self.levels[-1]:
                # Advance to the next level
                current_index = self.levels.index(self.current_level)
                self.current_level = self.levels[current_index + 1]
                logger.info(f"Curriculum level increased to {self.current_level}: {self.get_level_description()}")
                
                # Optionally, clear buffers to require re-qualification for the next level
                self.recent_rewards = []
                self.recent_complex_counts = []
            else:
                logger.info("Already at the maximum curriculum level.")

    def get_level_description(self):
        """
        Returns a description of the current curriculum level.

        Returns:
            str: Description of the current level.
        """
        descriptions = {
            1: "Basic molecules",
            2: "Intermediate molecules",
            2.5: "Advanced molecules",
            3: "Expert molecules"
        }
        return descriptions.get(self.current_level, "Unknown level")

    def get_current_level(self):
        """
        Returns the current curriculum level.

        Returns:
            float: Current curriculum level.
        """
        return self.current_level
