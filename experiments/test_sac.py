from stable_baselines3 import SAC

import slaythespire as sts
from environment.sts_env import StsFightEnv
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
import sys
import jax.numpy as jnp

import gymnasium as gym
import importlib

input_size = 20

checkpoint_dir = sys.argv[1]
model_name = sys.argv[2]

import_filename = f"data.train_{model_name}"
module = importlib.import_module(import_filename)

checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
model = nnx.eval_shape(lambda: module.Model(1, input_size, rngs=nnx.Rngs(0)))
_, _, abstract_state = nnx.split(model, nnx.RngState, ...)
state_restored = checkpointer.restore(f"{checkpoint_dir}/{model_name}", abstract_state)
nnx.update(model, state_restored)
model.eval()

def mock_embedding(card: int):
    return np.array([card/370.0], np.float32)

def embedding_wrapper(x):
    x = jnp.expand_dims(x, axis=1)
    x = model(x)
    return x

env = StsFightEnv(sts.CharacterClass.IRONCLAD, 0, 15, 5, embedding_wrapper, 20)
env = gym.wrappers.TimeLimit(env, 1000)
print(env.observation_space)
obs, _ = env.reset()
print(obs.shape)

model = SAC("MlpPolicy", env, verbose=1, device='cpu', tensorboard_log=f'ppo_log/{model_name}')
iterations = 36
for i in range(iterations):
	model.learn(total_timesteps=100_000, reset_num_timesteps=(i==0))
	model.save(f"sac_sts_{i}")


