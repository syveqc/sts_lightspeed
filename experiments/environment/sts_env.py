import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict
import time
from flax import nnx
import slaythespire as sts
import torch

from typing import Callable, Union

class StsFightEnv(gym.Env):
    def __init__(self, character_class: sts.CharacterClass, ascension: int, decksize: int, cards_from_start: int, embedding: Union[nnx.Module, torch.nn.Module, Callable[[int], np.typing.ArrayLike]], embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.observation_space = Box(-1, 1, (214+10*embedding_dim,))
        self.action_space = Box(-1,1,(embedding_dim+1,))

        self.character_class = character_class
        self.ascension = ascension

        self.decksize = decksize
        self.cards_from_start = cards_from_start

        self.monster_encounters = sts.RLInterface.getImplementedMonsterEncounters();

        self.embeddings = embedding(np.arange(371)) # type: ignore


    def reset(self, **kwargs):
        if 'seed' in kwargs and kwargs['seed'] is not None:
            seed = kwargs['seed']
        else:
            seed = int(time.time())

        # re-seed numpy
        np.random.seed(seed)

        # setup game
        self.gc = sts.GameContext(self.character_class, seed, self.ascension)
        self.gc.generateRandomDeck(self.decksize-self.cards_from_start, self.character_class, seed, self.cards_from_start)
        self.bc = sts.BattleContext()
        encounter = self.monster_encounters[np.random.randint(0, len(self.monster_encounters))]
        self.bc.init(self.gc, encounter)

        return self._getObservation(), {}

    def step(self, action):
        # take action in simulator
        card_to_play = self._getClosestCardHandIdx(action[:-1])
        target = int((action[-1]+1)*3)
        if target < 5:
            self.bc.playCardInHand(card_to_play, target)
        else:
            self.bc.endTurn()

        # observe new state and reward
        obs = self._getObservation()
        reward = 0
        if self.bc.outcome == sts.Outcome.PLAYER_VICTORY:
            reward = 1
        elif self.bc.outcome == sts.Outcome.PLAYER_LOSS:
            reward = -1
        terminated = self.bc.outcome != sts.Outcome.UNDECIDED

        return obs, reward, terminated, False, {}

    def _getHandEmbeddings(self) -> list[np.typing.ArrayLike]:
        return [np.array(self.embeddings[int(card.id)], np.float32) for card in self.bc.getCardsInHand()]  # type: ignore

    def _getClosestCardHandIdx(self, embedding_action: np.typing.ArrayLike) -> np.intp:
        return np.argmin([np.linalg.norm(embedding_action-card_embedding) for card_embedding in self._getHandEmbeddings()])  # type: ignore

    def _multisetAggregation(self, embeddings: list[np.typing.ArrayLike]) -> np.typing.ArrayLike:
        # TODO: look up how to do actual multiset aggregation and whether it is useful
        while len(embeddings) < 10:
            embeddings.append(np.zeros((self.embedding_dim,), np.float32))
        return np.concatenate(embeddings)

    def _getObservation(self):
        state = np.array(sts.RLInterface.getStateEmbedding(self.gc, self.bc))
        hand_embedding = self._multisetAggregation(self._getHandEmbeddings())
        return np.concatenate([state, hand_embedding])

if __name__ == '__main__':
    def mock_embedding(card: int):
        return np.array([card/370.0], np.float32)
    env = StsFightEnv(sts.CharacterClass.IRONCLAD, 0, 15, 5, mock_embedding, 1)

    # test space setup
    assert env.observation_space.shape == (224,), f"Observation space shape should be (224,), is {env.observation_space.shape}"
    assert env.action_space.shape == (2,), f"first part of the action space should have shape (2,), is {env.action_space.shape}"  # type: ignore
    # assert env.action_space['target'].n == 6, f'second part of the action space should be discrete of size 6, is {env.action_space['target'].n}'  # type: ignore

    # test random consistency and running without errors
    obs, _ = env.reset(seed=0)
    assert np.allclose(obs[4], 54.0, atol=1e-8), f"5th entry of obs should be 54.0, is {obs[4]}"
    assert np.allclose(obs[-6], 0.6, atol=1e-8), f"-6th entry of obs should be ~0.6, is {obs[-6]}"

    # for card in env.bc.getCardsInHand():
    #       print(f"{card}: {int(card.id)/370}")
    # sts.RLInterface.prettyPrintStateEmbedding(env.gc, env.bc)

    # test strike on invalid monster
    obs, reward, done, _, _ = env.step([0.86, -0.5])
    assert obs[-11]==5.0
    assert obs[5]==170.0
    assert reward == 0
    assert not done

    # test strike on monster
    obs, reward, done, _, _ = env.step([0.86, -1])
    assert obs[-11]==4.0
    assert obs[5]==164.0
    assert reward == 0
    assert not done

    # test ending round
    obs, reward, done, _, _ = env.step([0.2, 1])
    assert obs[-11]==5.0
    assert reward == 0
    assert not done

    # test taking damage
    obs, reward, done, _, _ = env.step([0.2, 1])
    assert obs[115]==48.0
    assert reward == 0
    assert not done

    # test dying
    obs, reward, done, _, _ = env.step([0.2, 1])
    obs, reward, done, _, _ = env.step([0.2, 1])
    assert done
    # sts.RLInterface.prettyPrintStateEmbedding(env.gc, env.bc)


