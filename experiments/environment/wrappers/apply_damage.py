from typing import SupportsFloat
import gymnasium as gym

from environment.sts_env import StsFightEnv

class ApplyDamageWrapper(gym.RewardWrapper):
    def __init__(self, env: StsFightEnv, coef=1.0):
        super().__init__(env)
        self.coef = coef

    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        hp_difference = 0.0
        for i in range(5):
            hp_difference += self.unwrapped.current_obs[4+22*i+1] - self.unwrapped.last_obs[4+22*i+1]  # type: ignore

        reward -= self.coef * hp_difference/100  # type: ignore

        return reward

