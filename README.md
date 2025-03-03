# sts_lightspeed for Reinforcement Learning

This is a modified version of the [sts_lightspeed](https://github.com/gamerpuppy/sts_lightspeed) repo with the goal of using Reinforcement Learning to train a Slay the Spire Agent with super-human performance in winrate.

## Getting Started

To get started, first make sure you have `cmake` installed on your system. ([CMake](https://cmake.org/))

Next, clone this repo and `cd` into it.
```bash
git clone --recurse-submodules https://github.com/syveqc/sts_lightspeed
cd sts_lightspeed
```

Then you can build it by running `cmake` (running in a `build` folder is recommended)
```bash
mkdir build
cd build
cmake ..
cmake --build .
```

## Importing the `python` interface

To use the python interface, make sure that the generated shared object (e.g. `slaythespire.cpython-312-x86_64-linux-gnu.so`) is accessable to `python`. (e.g. by adding the `build` folder to the `PYTHONPATH` environment variable or executing `python` inside the build folder)

Then you can simply import the module by running
```python
import slaythespire
```

All remaining sections using the `python` interface were tested using `python 3.12`. Also make sure the `requirements.txt` from the `experiments` folder was installed.


## Card Feature Embeddings

To generate the feature embeddings for the cards, first run 
```bash
cd experiments/data
python collect_card_data.py
```
to generate a dataset of card interactions.

Next, create the folders for saving the model and the training losses: (both names can be arbitrary, but then have to be adapted in the scripts below)
```bash
mkdir checkpoints
mkdir losses
```

Then you can train the embeddings you want from the scripts in the `experiments/data` folder, so for example to train the base features, you run
```bash
python train_base.py $(pwd)/checkpoints losses
```
giving the script the path to the folder for saving the model and name of the losses folder. (The folder for saving the model has to be given as an absolute path, for the losses either relative or absolute work.)


## Fighting Environment

To run training on the fighting environment, which contains training on random monster encounters with a random deck of cards, make sure you trained some feature embeddings like in the previous section, then run
```bash
cd experiments
python test_ppo.py $(pwd)/data/checkpoints base
```
where you can replace `base` with the name of the feature-model you trained.

Additionally, in `test_ppo.py`, you can pass the path to a `.json` file in the `StsFightEnv`, which has the following configuration options:
 - `monster_encounters`: list of integers or strings representing the `slaythespire.MonsterEncounterId`s of possible encounters (which is sampled uniformly from)
 - `ascension`: integer, the ascension to play at
 - `cards_from_start`: integer, how many starting cards are retained, e.g. if `cards_from_start: 8`, then two of the starting cards are randomly discarded for each battle
 - `decksize`: integer, how many cards the deck should contain, e.g. if `decksize: 15` and `cards_from_start: 10`, then all starting cards are retained and 5 random cards are sampled.
 - `starting_cards`: list of integers or strings, representing the `slaythespire.CardId`s of cards that should be added to the deck after generating the random deck. (so if `decksize: 15` and two `starting_cards` are given, the final decksize will be 17)
 - `curHp`: integer, the HP of the player for the fight
 - `maxHp`: integer, the maximum HP of the player for the fight


## Card Decision Environment

The card decision environment runs in conjunction with the fighting environment, letting the agent choose a card from three random cards after each battle. To run the two in conjunction start the server with
```bash
cd experiments
python environment/remote_fighting_env.py $(pwd)/data/checkpoints base
```

Once the server is started, run in a separate terminal
```bash
cd experiments
python environment/card_decision_env.py $(pwd)/data/checkpoints base
```

## Planned Features

 - [`gymnasium`](https://gymnasium.farama.org) interfaces for running Reinforcement Learning training on different stages of the game for the Ironclad (in this order):
    - Fights (done)
    - Card Selection (done)
    - Relic Selction
    - Path Selection
 - `gymnasium` interfaces should be compatible with each other, e.g. training on fights and card selection should be possible to run concurrently, i.e. the card selection agent should learn which cards are optimal to select for the current fighting agent.
 - training pipelines for those `gymnasium` environments.
