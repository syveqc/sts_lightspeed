from typing import SupportsFloat
import gymnasium as gym

from environment.sts_env import StsFightEnv

class UsedUpEnergyAtEnd(gym.RewardWrapper):
    def __init__(self, env: StsFightEnv, coef = 1.0):
        super().__init__(env)
        self.coef = coef

    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        energy_left = 3
        if self.unwrapped.end_turn:  # type: ignore
            energy_left = self.unwrapped.current_obs[117]  # type: ignore
            #print(f'energy reward: {3-min(energy_left,3)}')

        reward += self.coef * (3-min(energy_left,3))/3  # type: ignore

        return reward

