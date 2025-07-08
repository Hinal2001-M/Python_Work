# waymax_env.py
import gymnasium as gym
import numpy as np

class WaymaxRLWrapper(gym.Env):
    def __init__(self):
        super(WaymaxRLWrapper, self).__init__()

        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)


        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.ego = {'pos': np.array([0.0, 0.0]), 'heading': 0.0, 'speed': 5.0}
        self.nearby_agents = [self._gen_agent() for _ in range(5)]
        self.t = 0
        obs = self._get_obs()
        return obs, {}  # Gymnasium requires (obs, info)

    def step(self, action):
        reward, done, truncated, info = 0.0, False, False, {}

        if action == 0: self.ego['speed'] += 1.0
        elif action == 1: self.ego['speed'] -= 1.0
        elif action == 3: self.ego['speed'] = max(0.0, self.ego['speed'] - 5.0)
        elif action == 4: self.ego['speed'] = 0.0

        self.ego['speed'] = np.clip(self.ego['speed'], 0.0, 20.0)

        if self._collision():
            reward -= 10.0
            done = True
            info['collision'] = True

        self.t += 1
        if self.t >= 100:
            done = True
            reward += 20.0
            info['success'] = True  

        reward += 1.0  
        obs = self._get_obs()
        return obs, reward, done, truncated, info  

    def _gen_agent(self):
        return {
            'rel_pos': np.random.uniform(-30, 30, size=2),
            'rel_speed': np.random.uniform(-5, 5),
            'heading': np.random.uniform(-3.14, 3.14)
        }

    def _get_obs(self):
        ego_obs = [*self.ego['pos'], self.ego['speed'], self.ego['heading']]
        agents_obs = []
        for agent in self.nearby_agents:
            agents_obs.extend([*agent['rel_pos'], agent['rel_speed'], agent['heading']])
        return np.array(ego_obs + agents_obs[:20], dtype=np.float32)  # returns 24 values

    def _collision(self):
        for agent in self.nearby_agents:
            if np.linalg.norm(agent['rel_pos']) < 2.0:  # simple distance check
                return True
        return False
