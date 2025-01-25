import torch
import gymnasium as gym
import numpy as np
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
import torch.nn as nn

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


# Funkcja do ładowania modelu i uruchamiania w środowisku
def run_trained_model(model_path, env_name="BipedalWalker-v3", render=True):
    # Konfiguracja urządzenia
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Tworzenie środowiska
    env = gym.make(env_name, render_mode="human" if render else "rgb_array")
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]
    action_scale = 6  # Skala akcji (musi być taka sama jak przy treningu)

    # Tworzenie agenta i ładowanie wag modelu
    agent = BQN(state_space, action_space, action_scale, learning_rate=0.0001, device=device).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()  # Ustawienie modelu w trybie ewaluacji

    real_action = np.linspace(-1., 1., action_scale)  # Przeskalowane akcje
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Renderowanie środowiska
        if render:
            env.render()

        # Przewidywanie akcji za pomocą modelu
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).reshape(1, -1).to(device)
            action_prob = agent.action(state_tensor)
            action_indices = [int(x.max(1)[1].item()) for x in action_prob]
        selected_action = np.array([real_action[a] for a in action_indices])

        # Wykonanie akcji w środowisku
        next_state, reward, terminated, truncated, _ = env.step(selected_action)
        total_reward += reward
        done = terminated or truncated
        state = next_state

    print(f"Total Reward: {total_reward:.2f}")
    env.close()


# Ścieżka do wyuczonego modelu
model_path = "./model_weights/agent_500.pth"  # Dostosuj do swojej ścieżki
run_trained_model(model_path)
