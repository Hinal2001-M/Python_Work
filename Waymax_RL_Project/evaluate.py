# eval.py
from stable_baselines3 import PPO
from waymax_env import WaymaxRLWrapper
import numpy as np

env = WaymaxRLWrapper()
model = PPO.load("ppo_waymax_model")

episodes = 10
successes = 0
collisions = 0

for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        if reward < -5:
            collisions += 1
            break
    if reward > 0:
        successes += 1

print(f"Success rate: {successes}/{episodes}")
print(f"Collisions: {collisions}/{episodes}")










# from stable_baselines3 import PPO
# from waymax_env import WaymaxRLWrapper
# import numpy as np

# env = WaymaxRLWrapper("path/to/scenario.tfrecord")
# model = PPO.load("ppo_waymax")

# successes = 0
# collisions = 0
# episodes = 20

# for _ in range(episodes):
#     obs = env.reset()
#     done = False
#     while not done:
#         action, _ = model.predict(obs, deterministic=True)
#         obs, reward, done, info = env.step(action)

#         if info.get('collision'): collisions += 1
#         if info.get('success'): successes += 1

# print(f"Success Rate: {successes/episodes*100:.2f}%")
# print(f"Collision Rate: {collisions/episodes*100:.2f}%")
