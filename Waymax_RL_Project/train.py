from waymax_env import WaymaxRLWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

env = WaymaxRLWrapper()
check_env(env)  #should pass with no errors
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
model.save("ppo_waymax_model")
