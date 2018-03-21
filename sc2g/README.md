# sc2g

The SC2-Gym adapter (sc2g) converts the SC2 environment exposed by pysc2 into a OpenAI Gym environment so that algorithms from the `baselines` package can interact with the SC2 environment.

## Requirements
- Forked version of PySC2 2.0 from the `pysc2` folder in this repository.
- Baselines v0.1.5
- Gym v0.9.6

Make sure you are running our variant of PySC2 and not the version from the official repository!

Run `pip install -r requirements.txt` to install the correct versions of pysc2 and baselines.

## Installation
Run `pip install -e .` in this folder to make the sc2g package available to other Python scripts.