import gymnasium as gym
import numpy as np
import socket
import json

import slaythespire as sts

from typing import Callable, Union

import torch
from flax import nnx

from environment.sts_env import StsFightEnv

class RemoteFightingEnv(StsFightEnv):
    def __init__(self, character_class: sts.CharacterClass, ascension: int, embedding: Union[nnx.Module, torch.nn.Module, Callable[[int], np.typing.ArrayLike]], embedding_dim: int, ip: str, port: int):
        super().__init__(character_class, ascension, 10, 10, embedding, embedding_dim)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((ip, port))
        sock.listen()
        print(f'bound to {ip}:{port}, waiting for connections...')
        self.conn, self.addr = sock.accept()
        print(f'connection established to {self.addr}')
        self.timestep = 0
        self.data = None
        self.receive = True # make sure it is our turn to receive

        super().reset()

    def reset(self, **kwargs):
        print('reset')

        if self.receive:
            self.data = self.conn.recv(1024)
            self.receive = False
        data = json.loads(self.data.decode('utf-8'))
        print(data)
        self.gc.obtain_card(sts.Card(sts.CardId(data['card_id'])))
        kwargs['regenerate_deck'] = data['reset'] or self.bc.outcome == sts.Outcome.PLAYER_LOSS

        obs, info = super().reset(**kwargs)

        self.timestep = 0
        return obs, info

    def step(self, action): # type: ignore
        print('step')
        self.timestep += 1
        obs, reward, terminated, _, info = super().step(action)
        truncated = self.timestep > 1000

        if terminated or truncated:
            self.conn.send(json.dumps(self._getJsonObservation()).encode('utf-8'))
            self.receive = True

        return obs, reward, terminated, truncated, info

    def _getJsonObservation(self):
        return {'player': sts.RLInterface.getPlayerGameEmbedding(self.gc),
                'deck': [int(card.id) for card in self.gc.deck],
                'card_options': [int(card.id) for card in self.gc.get_card_rewards(self.encounter)],
                'terminated': self.gc.outcome != sts.GameOutcome.UNDECIDED}

if __name__ == '__main__':
    from stable_baselines3 import PPO

    import slaythespire as sts
    import numpy as np
    import orbax.checkpoint as ocp
    from flax import nnx
    import sys
    import jax.numpy as jnp

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

    env = RemoteFightingEnv(sts.CharacterClass.IRONCLAD, 0, embedding_wrapper, 20, 'localhost', 7351)
    obs, _ = env.reset()

    ppo = PPO("MlpPolicy", env, verbose=1, device='cpu', tensorboard_log=f"ppo_log/{model_name}")
    ppo.learn(total_timesteps=250000)
    ppo.save("ppo_fighting")


