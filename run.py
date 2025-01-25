import os
import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np

# Define necessary classes as per your original PPO implementation

class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        bias = self._bias.t().view(1, -1)
        return x + bias

class FixedNormal(torch.distributions.Normal):
    def log_probs(self, x):
        return super().log_prob(x).sum(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean

class DiagGaussian(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(DiagGaussian, self).__init__()
        self.fc_mean = nn.Linear(inp_dim, out_dim)
        self.b_logstd = AddBias(torch.zeros(out_dim))

    def forward(self, x):
        mean = self.fc_mean(x)
        logstd = self.b_logstd(torch.zeros_like(mean))
        return FixedNormal(mean, logstd.exp())

class PolicyNet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(PolicyNet, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(s_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.dist = DiagGaussian(128, a_dim)

    def forward(self, state, deterministic=False):
        feature = self.main(state)
        dist = self.dist(feature)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        return action, dist.log_probs(action)

    def choose_action(self, state, deterministic=False):
        with torch.no_grad():
            action, _ = self.forward(state, deterministic)
        return action

# Function to load the trained PolicyNet model

def load_trained_model(model_path, s_dim, a_dim, device):
    """
    Loads the trained PolicyNet model from the specified path.

    Args:
        model_path (str): Path to the saved model file.
        s_dim (int): Dimension of the state space.
        a_dim (int): Dimension of the action space.
        device (torch.device): Device to load the model on.

    Returns:
        PolicyNet: The loaded policy network.
    """
    policy_net = PolicyNet(s_dim, a_dim).to(device)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if "PolicyNet" in checkpoint:
            policy_net.load_state_dict(checkpoint["PolicyNet"])
            print("PolicyNet loaded successfully.")
        else:
            # If the checkpoint does not contain "PolicyNet", attempt to load the entire state_dict
            policy_net.load_state_dict(checkpoint)
            print("Model loaded successfully (entire state_dict).")
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    policy_net.eval()
    return policy_net

# Function to run a single episode using the trained policy

def run_episode(env, policy_net, device, render=True):
    """
    Runs a single episode using the trained policy network.

    Args:
        env (gym.Env): The Gym environment.
        policy_net (PolicyNet): The trained policy network.
        device (torch.device): Device to run the computations on.
        render (bool): Whether to render the environment.

    Returns:
        float: Total reward accumulated in the episode.
        int: Length of the episode.
    """
    state, _ = env.reset()
    total_reward = 0.0
    length = 0

    while True:
        if render:
            env.render()

        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action = policy_net.choose_action(state_tensor, deterministic=True).cpu().numpy()[0]

        # Ensure action is within the action space bounds
        action = np.clip(action, env.action_space.low, env.action_space.high)

        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        length += 1

        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward:.2f}, Length: {length}")
            break

    return total_reward, length

# Main function to load the model and run the episode

def main():
    # Configuration
    MODEL_PATH = "save/model.pt"  # Path to your trained model
    ENV_NAME = "BipedalWalker-v3"
    RENDER = True  # Set to False to disable rendering

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize environment
    env = gym.make(ENV_NAME, render_mode='human')  # Use 'rgb_array' if 'human' causes issues
    state, _ = env.reset()
    s_dim = state.shape[0]
    a_dim = env.action_space.shape[0]
    print(f"State Dimension: {s_dim}, Action Dimension: {a_dim}")

    # Load the trained model
    try:
        policy_net = load_trained_model(MODEL_PATH, s_dim, a_dim, device)
    except FileNotFoundError as e:
        print(e)
        return
    except KeyError as e:
        print(f"Key error while loading the model: {e}")
        return
    except Exception as e:
        print(f"Unexpected error while loading the model: {e}")
        return

    # Run a single episode
    total_reward, length = run_episode(env, policy_net, device, render=RENDER)

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
