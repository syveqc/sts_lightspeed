from stable_baselines3 import PPO

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
ppo_name = sys.argv[3]

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

env = StsFightEnv(sts.CharacterClass.IRONCLAD, 0, 15, 5, embedding_wrapper, 20, print_flag=True)
env = gym.wrappers.TimeLimit(env, 1000)

model = PPO.load(ppo_name)

obs, info = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
