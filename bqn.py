import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import collections
import random
import numpy as np
import os
import gymnasium as gym
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class ReplayBuffer:
    def __init__(self, buffer_limit, action_space, device):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.action_space = action_space
        self.device = device

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        state_lst, reward_lst, next_state_lst, done_mask_lst = [], [], [], []
        actions_lst = [[] for _ in range(self.action_space)]

        for transition in mini_batch:
            state, actions, reward, next_state, done_mask = transition
            state_lst.append(state)
            for idx in range(self.action_space):
                actions_lst[idx].append(actions[idx])
            reward_lst.append([reward])
            next_state_lst.append(next_state)
            done_mask_lst.append([done_mask])

        state_array = np.array(state_lst, dtype=np.float32)
        reward_array = np.array(reward_lst, dtype=np.float32)
        next_state_array = np.array(next_state_lst, dtype=np.float32)
        done_mask_array = np.array(done_mask_lst, dtype=np.float32)

        states = torch.from_numpy(state_array).to(self.device)
        rewards = torch.from_numpy(reward_array).to(self.device)
        next_states = torch.from_numpy(next_state_array).to(self.device)
        done_masks = torch.from_numpy(done_mask_array).to(self.device)
        actions = [torch.tensor(x, dtype=torch.float32).to(self.device) for x in actions_lst]

        return (states, actions, rewards, next_states, done_masks)

    def size(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, state_space: int, action_num: int, action_scale: int):
        super(QNetwork, self).__init__()
        self.linear_1 = nn.Linear(state_space, state_space * 20)
        self.linear_2 = nn.Linear(state_space * 20, state_space * 10)
        self.actions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_space * 10, state_space * 5),
                nn.ReLU(),
                nn.Linear(state_space * 5, action_scale)
            ) for _ in range(action_num)
        ])
        self.value = nn.Sequential(
            nn.Linear(state_space * 10, state_space * 5),
            nn.ReLU(),
            nn.Linear(state_space * 5, 1)
        )

    def forward(self, x):
        x = F.relu(self.linear_1(x))
        encoded = F.relu(self.linear_2(x))
        actions = [action_head(encoded) for action_head in self.actions]
        value = self.value(encoded)
        for i in range(len(actions)):
            actions[i] = actions[i] - actions[i].max(-1, keepdim=True)[0]
            actions[i] += value
        return actions

class BQN(nn.Module):
    def __init__(self, state_space: int, action_num: int, action_scale: int, learning_rate: float, device: str):
        super(BQN, self).__init__()
        self.q = QNetwork(state_space, action_num, action_scale).to(device)
        self.target_q = QNetwork(state_space, action_num, action_scale).to(device)
        self.target_q.load_state_dict(self.q.state_dict())
        self.optimizer = optim.Adam([
            {'params': self.q.linear_1.parameters(), 'lr': learning_rate / (action_num + 2)},
            {'params': self.q.linear_2.parameters(), 'lr': learning_rate / (action_num + 2)},
            {'params': self.q.value.parameters(), 'lr': learning_rate / (action_num + 2)},
            {'params': self.q.actions.parameters(), 'lr': learning_rate},
        ])
        self.update_freq = 1000
        self.update_count = 0
        self.device = device

    def action(self, x):
        return self.q(x)

    def train_mode(self, n_epi, memory, batch_size, gamma, use_tensorboard, writer):
        states, actions, rewards, next_states, done_masks = memory.sample(batch_size)
        actions = torch.stack(actions).transpose(0, 1).unsqueeze(-1)
        done_masks = 1 - done_masks

        cur_actions = self.q(states)
        cur_actions = torch.stack(cur_actions).transpose(0, 1)
        cur_actions = cur_actions.gather(2, actions.long()).squeeze(-1)

        target_cur_actions = self.target_q(next_states)
        target_cur_actions = torch.stack(target_cur_actions).transpose(0, 1)
        target_cur_actions = target_cur_actions.max(-1, keepdim=True)[0]
        target_action = (done_masks * gamma * target_cur_actions.mean(1) + rewards)

        loss = F.mse_loss(cur_actions, target_action.repeat(1, cur_actions.shape[1]))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        if (self.update_count % self.update_freq == 0) and (self.update_count > 0):
            self.update_count = 0
            self.target_q.load_state_dict(self.q.state_dict())

        if use_tensorboard and writer is not None:
            writer.add_scalar("Loss/loss", loss.item(), n_epi)

        return loss.item()

def main():
    TRAIN = True
    RENDER = False
    EPOCHS = 2000
    USE_TENSORBOARD = True
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 64
    GAMMA = 0.99
    ACTION_SCALE = 6
    SAVE_INTERVAL = 100
    PRINT_INTERVAL = 1
    LOAD_MODEL = 'no'

    if USE_TENSORBOARD:
        writer = SummaryWriter()
    else:
        writer = None

    os.makedirs('./model_weights', exist_ok=True)
    env = gym.make("BipedalWalker-v3", render_mode="human" if RENDER else "rgb_array")
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = BQN(state_space, action_space, ACTION_SCALE, LEARNING_RATE, device).to(device)
    if LOAD_MODEL != 'no':
        agent.load_state_dict(torch.load(f'./model_weights/{LOAD_MODEL}', map_location=device))
    memory = ReplayBuffer(buffer_limit=100000, action_space=action_space, device=device)
    real_action = np.linspace(-1., 1., ACTION_SCALE)
    scores = []
    avg_scores = []
    for n_epi in range(1, EPOCHS + 1):
        state, info = env.reset()
        done = False
        score = 0.0
        while not done:
            if RENDER:
                env.render()
            epsilon = max(0.01, 0.9 - 0.01 * (n_epi / 10))
            if epsilon > random.random():
                action_indices = random.sample(range(ACTION_SCALE), action_space)
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).reshape(1, -1).to(device)
                    action_prob = agent.action(state_tensor)
                    action_indices = [int(x.max(1)[1].item()) for x in action_prob]
            selected_action = np.array([real_action[a] for a in action_indices])
            next_state, reward, terminated, truncated, _ = env.step(selected_action)
            done = terminated or truncated
            score += reward
            done_mask = 0 if not done else 1
            memory.put((state, action_indices, reward, next_state, done_mask))
            state = next_state
            if memory.size() > 5000 and TRAIN:
                loss = agent.train_mode(n_epi, memory, BATCH_SIZE, GAMMA, USE_TENSORBOARD, writer)
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        if USE_TENSORBOARD and writer is not None:
            writer.add_scalar("Reward/score", score, n_epi)
            writer.add_scalar("Reward/avg_score", avg_score, n_epi)
        if (n_epi % SAVE_INTERVAL == 0) and (n_epi > 0):
            torch.save(agent.state_dict(), f'./model_weights/agent_{n_epi}.pth')
        if (n_epi % PRINT_INTERVAL == 0):
            print(f"Episode: {n_epi}, Score: {score:.2f}, Average Score: {avg_score:.2f}, Epsilon: {epsilon:.2f}")
    plt.figure(figsize=(12, 5))
    plt.plot(scores, label='Score per Episode')
    plt.plot(avg_scores, label='Average Score (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.show()
    if writer is not None:
        writer.close()
    env.close()

if __name__ == "__main__":
    main()
