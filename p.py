import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.callbacks import BaseCallback
import torch
import matplotlib.pyplot as plt
import os
import ale_py

gym.register_envs(ale_py)

class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'episode' in info.keys():
                reward = info['episode']['r']
                self.episode_rewards.append(reward)
                if self.verbose > 0:
                    print(f"Epizod {len(self.episode_rewards)}: Nagroda = {reward}")
        return True

env_id = "ALE/Tennis-v5"
num_envs = 4
env = make_atari_env(env_id, n_envs=num_envs, seed=0)
env = VecTransposeImage(env)
env = VecFrameStack(env, n_stack=4)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Używany device: {device}")

model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="./ppo_tennis_tensorboard/",
    device=device
)

reward_callback = RewardCallback(verbose=1)
time_steps = 1_000_000
model.learn(total_timesteps=time_steps, callback=reward_callback, log_interval=10)
model.save("ppo_tennis")
env.close()

plt.figure(figsize=(12, 6))
plt.plot(reward_callback.episode_rewards, label='Nagroda Epizodu')
window = 100
if len(reward_callback.episode_rewards) >= window:
    smoothed_rewards = pd.Series(reward_callback.episode_rewards).rolling(window=window).mean()
    plt.plot(smoothed_rewards, label=f'Średnia krocząca ({window} epizodów)')
plt.xlabel('Numer Epizodu')
plt.ylabel('Nagroda')
plt.title('Nagrody z Epizodów podczas Treningu')
plt.legend()
plt.grid(True)
plt.show()

eval_env = gym.make(env_id)
model = PPO.load("ppo_tennis")

episodes = 5

for episode in range(1, episodes + 1):
    obs, _ = eval_env.reset()
    done = False
    total_reward = 0
    while not done:
        eval_env.render()
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
        total_reward += reward
    print(f"Epizod {episode}: Całkowita nagroda = {total_reward}")

eval_env.close()
