from typing import SupportsFloat
import gymnasium as gym

from environment.sts_env import StsFightEnv

class StrengthWrapper(gym.RewardWrapper):
    def __init__(self, env: StsFightEnv, coef=1.0):
        super().__init__(env)
        self.coef = coef

    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        strength_difference = self.unwrapped.current_obs[126] - self.unwrapped.last_obs[126]  # type: ignore

        reward += self.coef * strength_difference/10  # type: ignore

        return reward

