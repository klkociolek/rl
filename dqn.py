import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from PIL import Image
from torchvision import transforms
import ale_py

gym.register_envs(ale_py)


env = gym.make('ALETennis-v5', render_mode='rgb_array')
state_shape = (4, 84, 84)
n_actions = env.action_space.n


gamma = 0.99
epsilon_start = 1.0
epsilon_min = 0.1
epsilon_decay = 0.999
batch_size = 32
buffer_size = 100000
target_update = 1000
learning_rate = 0.00025
max_episodes = 1000
max_steps = 5000

# Preprocessing transform
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((84, 84)),
    transforms.ToTensor(),
])


class DQN(nn.Module)
    def __init__(self, in_channels, num_actions)
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64  7  7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ReplayBuffer
    def __init__(self, capacity)
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size)
        return random.sample(self.buffer, batch_size)

    def __len__(self)
        return len(self.buffer)


class DQNAgent
    def __init__(self, in_channels, num_actions)
        self.device = torch.device(cuda if torch.cuda.is_available() else cpu)
        self.policy_net = DQN(in_channels, num_actions).to(self.device)
        self.target_net = DQN(in_channels, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)
        self.epsilon = epsilon_start
        self.steps = 0

    def preprocess_state(self, state)
        img = Image.fromarray(state)
        return transform(img).unsqueeze(0)

    def get_state(self, state)
        if not hasattr(self, 'state_buffer')
            # Initialize state buffer with 4 identical frames
            self.state_buffer = torch.cat([self.preprocess_state(state)]  4, dim=1)
        else
            # Add new frame and remove oldest
            self.state_buffer = torch.cat(
                [self.state_buffer[, 1], self.preprocess_state(state)], dim=1
            )
        return self.state_buffer

    def select_action(self, state)
        if random.random()  self.epsilon
            return torch.tensor([[random.randrange(n_actions)]], device=self.device)
        else
            with torch.no_grad()
                return self.policy_net(state.to(self.device)).max(1)[1].view(1, 1)

    def update_epsilon(self)
        self.epsilon = max(epsilon_min, self.epsilon  epsilon_decay)

    def update_model(self)
        if len(self.memory)  batch_size
            return

        transitions = self.memory.sample(batch_size)
        batch = list(zip(transitions))

        state_batch = torch.cat(batch[0]).to(self.device)
        action_batch = torch.cat(batch[1]).to(self.device)
        reward_batch = torch.cat(batch[2]).to(self.device)
        next_state_batch = torch.cat(batch[3]).to(self.device)
        done_batch = torch.cat(batch[4]).to(self.device)

        q_values = self.policy_net(state_batch).gather(1, action_batch)

        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + gamma  next_q_values  (1 - done_batch)

        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % target_update == 0
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self, env, max_episodes)
        episode_rewards = []

        for episode in range(max_episodes)
            state, _ = env.reset()
            state = self.get_state(state)
            total_reward = 0
            done = False

            while not done
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated

                next_state_tensor = self.get_state(next_state)
                self.memory.push(
                    state,
                    action,
                    torch.tensor([reward], device=self.device),
                    next_state_tensor,
                    torch.tensor([done], dtype=torch.float, device=self.device)
                )

                state = next_state_tensor
                total_reward += reward

                self.update_model()
                self.update_epsilon()

            episode_rewards.append(total_reward)
            print(fEpisode {episode + 1}, Total Reward {total_reward.2f}, Epsilon {self.epsilon.3f})

            if (episode + 1) % 100 == 0
                torch.save(self.policy_net.state_dict(), fdqn_tennis_{episode + 1}.pth)

        return episode_rewards


if __name__ == __main__
    agent = DQNAgent(state_shape[0], n_actions)
    rewards = agent.train(env, max_episodes)
    env.close()