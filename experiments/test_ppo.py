from stable_baselines3 import PPO

import slaythespire as sts
from environment.sts_env import StsFightEnv
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
import sys
import jax.numpy as jnp

import gymnasium as gym

class Model(nnx.Module):
    def __init__(self, din, dout, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(din, 512, rngs=rngs)
        self.batch_norm1 = nnx.BatchNorm(512, rngs=rngs)
        self.dropout1 = nnx.Dropout(0.2, rngs=rngs)
        self.linear2 = nnx.Linear(512, 512, rngs=rngs)
        self.linear3 = nnx.Linear(512, 512, rngs=rngs)
        self.linear4 = nnx.Linear(512, 512, rngs=rngs)
        self.linear5 = nnx.Linear(512, 512, rngs=rngs)
        self.linear6 = nnx.Linear(512, dout, rngs=rngs)

    def __call__(self, x):
        x = jnp.expand_dims(jnp.array(x, dtype=jnp.float32), 1)
        x = nnx.relu(self.linear1(x))
        x = self.dropout1(self.batch_norm1(x))
        x = nnx.relu(self.linear2(x))
        x = nnx.relu(self.linear3(x))
        x = nnx.relu(self.linear4(x))
        x = nnx.relu(self.linear5(x))
        return self.linear6(x)  # type: ignore

input_size = 20

checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
model = nnx.eval_shape(lambda: Model(1, input_size, rngs=nnx.Rngs(0)))
_, _, abstract_state = nnx.split(model, nnx.RngState, ...)
state_restored = checkpointer.restore(f"{sys.argv[1]}/pure_dict", abstract_state)
nnx.update(model, state_restored)
model.eval()

def mock_embedding(card: int):
    return np.array([card/370.0], np.float32)

env = StsFightEnv(sts.CharacterClass.IRONCLAD, 0, 15, 5, model, 20)
env = gym.wrappers.TimeLimit(env, 1000)
print(env.observation_space)
obs, _ = env.reset()
print(obs.shape)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=250000)
model.save("ppo_sts")


