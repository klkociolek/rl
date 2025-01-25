import os
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


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

    def evaluate(self, state, action):
        feature = self.main(state)
        dist = self.dist(feature)
        return dist.log_probs(action), dist.entropy()


class ValueNet(nn.Module):
    def __init__(self, s_dim):
        super(ValueNet, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(s_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        return self.main(state)[:, 0]


class EnvRunner:
    def __init__(self, s_dim, a_dim, gamma=0.99, lamb=0.95, max_step=2048, device='cpu'):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.gamma = gamma
        self.lamb = lamb
        self.max_step = max_step
        self.device = device
        self.mb_states = np.zeros((self.max_step, self.s_dim), dtype=np.float32)
        self.mb_actions = np.zeros((self.max_step, self.a_dim), dtype=np.float32)
        self.mb_values = np.zeros((self.max_step,), dtype=np.float32)
        self.mb_rewards = np.zeros((self.max_step,), dtype=np.float32)
        self.mb_a_logps = np.zeros((self.max_step,), dtype=np.float32)

    def compute_discounted_return(self, rewards, last_value):
        returns = np.zeros_like(rewards)
        n_step = len(rewards)

        for t in reversed(range(n_step)):
            if t == n_step - 1:
                returns[t] = rewards[t] + self.gamma * last_value
            else:
                returns[t] = rewards[t] + self.gamma * returns[t + 1]

        return returns

    def compute_gae(self, rewards, values, last_value):
        advs = np.zeros_like(rewards)
        n_step = len(rewards)
        last_gae_lam = 0.0

        for t in reversed(range(n_step)):
            if t == n_step - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value - values[t]
            advs[t] = last_gae_lam = delta + self.gamma * self.lamb * last_gae_lam

        return advs + values

    def run(self, env, policy_net, value_net):
        state, _ = env.reset()
        episode_len = self.max_step

        for step in range(self.max_step):
            state_tensor = torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float32, device=self.device)
            action, a_logp = policy_net(state_tensor)
            value = value_net(state_tensor)

            action = action.cpu().numpy()[0]
            a_logp = a_logp.cpu().numpy()
            value = value.cpu().numpy()

            self.mb_states[step] = state
            self.mb_actions[step] = action
            self.mb_a_logps[step] = a_logp
            self.mb_values[step] = value

            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            self.mb_rewards[step] = reward

            if done:
                episode_len = step + 1
                break
            state = observation

        last_value = value_net(
            torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float32, device=self.device)
        ).cpu().numpy()

        mb_returns = self.compute_discounted_return(self.mb_rewards[:episode_len], last_value)
        return self.mb_states[:episode_len], \
               self.mb_actions[:episode_len], \
               self.mb_a_logps[:episode_len], \
               self.mb_values[:episode_len], \
               mb_returns, \
               self.mb_rewards[:episode_len]


class PPO:
    def __init__(self, policy_net, value_net, lr=1e-4, max_grad_norm=0.5, ent_weight=0.01, clip_val=0.2,
                 sample_n_epoch=4, sample_mb_size=64, device='cpu'):
        self.policy_net = policy_net
        self.value_net = value_net
        self.max_grad_norm = max_grad_norm
        self.ent_weight = ent_weight
        self.clip_val = clip_val
        self.sample_n_epoch = sample_n_epoch
        self.sample_mb_size = sample_mb_size
        self.device = device
        self.opt_policy = torch.optim.Adam(policy_net.parameters(), lr)
        self.opt_value = torch.optim.Adam(value_net.parameters(), lr)

    def train(self, mb_states, mb_actions, mb_old_values, mb_advs, mb_returns, mb_old_a_logps):
        mb_states = torch.from_numpy(mb_states).to(self.device)
        mb_actions = torch.from_numpy(mb_actions).to(self.device)
        mb_old_values = torch.from_numpy(mb_old_values).to(self.device)
        mb_advs = torch.from_numpy(mb_advs).to(self.device)
        mb_returns = torch.from_numpy(mb_returns).to(self.device)
        mb_old_a_logps = torch.from_numpy(mb_old_a_logps).to(self.device)
        episode_length = len(mb_states)
        rand_idx = np.arange(episode_length)
        sample_n_mb = episode_length // self.sample_mb_size

        if sample_n_mb <= 0:
            sample_mb_size = episode_length
            sample_n_mb = 1
        else:
            sample_mb_size = self.sample_mb_size

        for _ in range(self.sample_n_epoch):
            np.random.shuffle(rand_idx)

            for j in range(sample_n_mb):
                sample_idx = rand_idx[j * sample_mb_size: (j + 1) * sample_mb_size]
                sample_states = mb_states[sample_idx]
                sample_actions = mb_actions[sample_idx]
                sample_old_values = mb_old_values[sample_idx]
                sample_advs = mb_advs[sample_idx]
                sample_returns = mb_returns[sample_idx]
                sample_old_a_logps = mb_old_a_logps[sample_idx]

                sample_a_logps, sample_ents = self.policy_net.evaluate(sample_states, sample_actions)
                sample_values = self.value_net(sample_states)
                ent = sample_ents.mean()

                v_pred_clip = sample_old_values + torch.clamp(sample_values - sample_old_values, -self.clip_val,
                                                              self.clip_val)
                v_loss1 = (sample_returns - sample_values).pow(2)
                v_loss2 = (sample_returns - v_pred_clip).pow(2)
                v_loss = torch.max(v_loss1, v_loss2).mean()

                ratio = (sample_a_logps - sample_old_a_logps).exp()
                pg_loss1 = -sample_advs * ratio
                pg_loss2 = -sample_advs * torch.clamp(ratio, 1.0 - self.clip_val, 1.0 + self.clip_val)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean() - self.ent_weight * ent

                self.opt_policy.zero_grad()
                pg_loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                self.opt_policy.step()

                self.opt_value.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.opt_value.step()

        return pg_loss.item(), v_loss.item(), ent.item()


def play(policy_net, env_eval, device, render=True):
    state, _ = env_eval.reset()
    total_reward = 0
    length = 0

    while True:
        if render:
            env_eval.render()

        state_tensor = torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float32, device=device)
        action = policy_net.choose_action(state_tensor, deterministic=True).cpu().numpy()
        observation, reward, terminated, truncated, info = env_eval.step(action[0])
        done = terminated or truncated
        total_reward += reward
        length += 1

        if done:
            print(f"[Evaluation] Total reward = {total_reward:.2f}, length = {length}")
            break
        state = observation


def train_agent(env_train, env_eval, runner, policy_net, value_net, agent, max_episode=5000):
    mean_total_reward = 0
    mean_length = 0
    save_dir = 'save'
    os.makedirs(save_dir, exist_ok=True)
    rewards_history = []
    lengths_history = []

    for i in range(1, max_episode + 1):
        with torch.no_grad():
            mb_states, mb_actions, mb_old_a_logps, mb_values, mb_returns, mb_rewards = runner.run(env_train, policy_net,
                                                                                                  value_net)
            mb_advs = mb_returns - mb_values
            mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-6)

        pg_loss, v_loss, ent = agent.train(mb_states, mb_actions, mb_values, mb_advs, mb_returns, mb_old_a_logps)
        mean_total_reward += mb_rewards.sum()
        mean_length += len(mb_states)
        rewards_history.append(mb_rewards.sum())
        lengths_history.append(len(mb_states))
        print(f"[Episode {i:5d}] Total Reward = {mb_rewards.sum():.2f}, Length = {len(mb_states)}")

        if i % 200 == 0:
            avg_reward = mean_total_reward / 200
            avg_length = mean_length / 200
            print(f"\n[{i:5d} / {max_episode:5d}]")
            print("----------------------------------")
            print(f"Actor Loss   = {pg_loss:.6f}")
            print(f"Critic Loss  = {v_loss:.6f}")
            print(f"Entropy      = {ent:.6f}")
            print(f"Mean Reward  = {avg_reward:.2f}")
            print(f"Mean Length  = {avg_length:.2f}")
            print("Saving the model...", end=" ")
            torch.save({
                "it": i,
                "PolicyNet": policy_net.state_dict(),
                "ValueNet": value_net.state_dict()
            }, os.path.join(save_dir, "model.pt"))
            print("Done.\n")
            print("Evaluating the agent:")
            play(policy_net, env_eval, device, render=True)
            mean_total_reward = 0
            mean_length = 0

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards_history)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')

    plt.subplot(1, 2, 2)
    plt.plot(lengths_history)
    plt.title('Episode Length')
    plt.xlabel('Episode')
    plt.ylabel('Length')

    plt.tight_layout()
    plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

env_train = gym.make('BipedalWalker-v3', render_mode='rgb_array')
env_eval = gym.make('BipedalWalker-v3', render_mode='human')

s_dim = env_train.observation_space.shape[0]
a_dim = env_train.action_space.shape[0]
print(f"State Dimension: {s_dim}")
print(f"Action Dimension: {a_dim}")

policy_net = PolicyNet(s_dim, a_dim).to(device)
value_net = ValueNet(s_dim).to(device)
runner = EnvRunner(s_dim, a_dim, device=device)
agent = PPO(policy_net, value_net, device=device)

print("Ocena agenta przed treningiem:")
play(policy_net, env_eval, device, render=True)

train_agent(env_train, env_eval, runner, policy_net, value_net, agent, max_episode=1000)

env_train.close()
env_eval.close()
