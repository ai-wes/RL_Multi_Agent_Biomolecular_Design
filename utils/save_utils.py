import torch
import os




def save_checkpoint(state, filename):
    print(f"Saving checkpoint to {filename}")
    print(f"Checkpoint contents: {state.keys()}")
    torch.save(state, filename)
    print(f"Checkpoint saved successfully")



def load_checkpoint(filename, agent, optimizer):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        agent.model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_episode = checkpoint['episode']
        print(f"Loaded checkpoint '{filename}' (episode {start_episode})")
        return start_episode
    else:
        print(f"No checkpoint found at {filename}")
        return 0
