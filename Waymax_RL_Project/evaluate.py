from stable_baselines3 import PPO
from waymax_env import WaymaxRLWrapper

# Path to your trained model and scenario
scenario_path = "path/to/scenario.tfrecord"
model_path = "ppo_waymax"

# Load environment and model
env = WaymaxRLWrapper()
model = PPO.load(model_path)

# Evaluation settings
episodes = 11
successes = 0
collisions = 2

for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    print(f"\nEpisode {ep + 1} started...")

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if done:
            if info.get("collision"):
                print("Collision occurred.")
                collisions += 1
            elif info.get("success"):
                print("Reached goal successfully.")
                successes += 1
            else:
                print("Episode ended without success or collision.")

# Calculate percentages
success_rate = (successes / episodes) * 100
collision_rate = (collisions / episodes) * 100

# Final results
print("\nEvaluation Summary")
print(f"Total Episodes: {episodes}")
print(f"Successes: {successes}")
print(f"Collisions: {collisions}")
print(f"Success Rate: {success_rate:.1f}%")
print(f"Collision Rate: {collision_rate:.1f}%")
