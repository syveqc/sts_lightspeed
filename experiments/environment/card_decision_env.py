import json
import socket
import time

import gymnasium as gym
import numpy as np
import slaythespire as sts

class CardDecisionEnv(gym.Env):
    def __init__(self, character_class: sts.CharacterClass, ascension: int, embedding, embedding_size, fighting_env_ip, fighting_env_port, max_deck_size=50):
        self.observation_space = gym.spaces.Dict({'player': gym.spaces.Box(-1, 1, shape=(3,)),
                                                  'deck': gym.spaces.Box(-1, 1, shape=(max_deck_size, embedding_size)),
                                                  'card_options': gym.spaces.Box(-1, 1, shape=(4, embedding_size))})
        self.action_space = gym.spaces.Box(-1, 1, shape=(embedding_size,))

        self.embeddings = embedding(np.arange(371)) # type: ignore
        self.embedding_size = embedding_size
        self.character_class = character_class
        self.ascension = ascension
        self.max_deck_size = max_deck_size

        self.fighting_env_ip = fighting_env_ip
        self.fighting_env_port = fighting_env_port

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.fighting_env_ip, self.fighting_env_port))
        self.current_card_options = None

        self.receive = False


    def reset(self, **kwargs):
        if 'seed' in kwargs and kwargs['seed'] is not None:
            seed = kwargs['seed']
        else:
            seed = int(time.time())

        self.gc = sts.GameContext(self.character_class, seed, self.ascension)

        data = {'card_id': 1, 'reset': True}
        self.socket.send(json.dumps(data).encode('utf-8'))

        obs = self._getObservation()[0]

        return obs, {}


    def step(self, action):
        self._obtainCard(action)
        obs, terminated = self._getObservation()
        reward = 1

        return obs, reward, terminated, False, {}

    def _obtainCard(self, action):
        card_id = self._getClosestCardIdToCurrentOptions(action)
        data = {'card_id': card_id, 'reset': False}
        self.socket.send(json.dumps(data).encode('utf-8'))

    def _getObservation(self):
        data = self.socket.recv(50000)
        data = json.loads(data.decode('utf-8'))
        self.current_card_options = data['card_options']
        deck_embeddings = [self.embeddings[cardid] for cardid in data['deck']]
        concatenated_deck = np.stack(deck_embeddings)
        padded_deck = np.pad(concatenated_deck, ((0, self.max_deck_size-len(data['deck'])), (0, 0)), 'constant')
        current_card_option_embeddings = [self.embeddings[cardid] for cardid in self.current_card_options]
        current_card_options_concat = np.stack(current_card_option_embeddings)
        padded_current_card_options = np.pad(current_card_options_concat, ((0, 4-len(self.current_card_options)), (0, 0)), 'constant')
        obs = {'player': np.array(data['player']),
               'deck': padded_deck,
               'card_options': padded_current_card_options}
        terminated = data['terminated'] or len(obs['deck']) > 50
        return obs, terminated

    def _getClosestCardIdToCurrentOptions(self, action):
        return self.current_card_options[np.argmin([np.linalg.norm(self.embeddings[cardid] - np.array(action)) for cardid in self.current_card_options])]

if __name__ == '__main__':
    from stable_baselines3 import PPO

    import slaythespire as sts
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

    env = CardDecisionEnv(sts.CharacterClass.IRONCLAD, 0, embedding_wrapper, 20, 'localhost', 7351)
    obs, _ = env.reset()

    ppo = PPO("MultiInputPolicy", env, verbose=1, device='cpu', tensorboard_log=f"ppo_log/{model_name}")
    ppo.learn(total_timesteps=250000)
    ppo.save("ppo_decisions")


