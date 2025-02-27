from typing import SupportsFloat
import gymnasium as gym

from environment.sts_env import StsFightEnv

class TakeDamageWrapper(gym.RewardWrapper):
    def __init__(self, env: StsFightEnv, coef = 1.0):
        super().__init__(env)
        self.coef = coef

    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        hp_difference = self.env.current_obs[115] - self.env.last_obs[115]  # type: ignore

        reward += self.coef * hp_difference/80

        return reward

