# sts_lightspeed for Reinforcement Learning

This is a modified version of the [sts_lightspeed](https://github.com/gamerpuppy/sts_lightspeed) repo with the goal of using Reinforcement Learning to train a Slay the Spire Agent with super-human performance in winrate.

## Getting Started

To get started, first make sure you have `cmake` installed on your system. ([CMake](https://cmake.org/))

Next, clone this repo and `cd` into it.
```bash
git clone https://github.com/syveqc/sts_lightspeed
cd sts_lightspeed
```

Then you can build it by running `cmake` (running in a `build` folder is recommended)
```bash
mkdir build
cd build
cmake ..
cmake -build .
```

## Running the `python` interface

To run the python interface, make sure that the generated shared object (e.g. `slaythespire.cpython-310-x86_64-linux-gnu.so`) is accessable to `python`. (e.g. by adding the `build` folder to the `PYTHONPATH` environment variable or executing `python` inside the build folder)

Then you can simply import the module by running
```python
import slaythespire
```

## Planned Features

 - [`gymnasium`](https://gymnasium.farama.org) interfaces for running Reinforcement Learning training on different stages of the game for the Ironclad (in this order):
    - Fights
    - Card Selection
    - Relic Selction
    - Path Selection
 - `gymnasium` interfaces should be compatible with each other, e.g. training on fights and card selection should be possible to run concurrently, i.e. the card selection agent should learn which cards are optimal to select for the current fighting agent.
 - training pipelines for those `gymnasium` environments.
