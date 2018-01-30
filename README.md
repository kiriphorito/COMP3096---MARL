# COMP3096 - Research Methods
Thanks to [chris-chris](https://github.com/chris-chris/pysc2-examples) and [Siraj Raval](https://github.com/Zacharias030/A-Guide-to-DeepMinds-StarCraft-AI-Environment)

## Overview

This is the code for [this](https://youtu.be/URWXG5jRB-A) video on on Youtube by Siraj Raval. This code will help you train or run a pretrained AI model in the DeepMind Starcraft II environment.

The following doesn't work with Python 2. You will require Python 3. So run all commands with pip3 and python3. If there are any problems in the installation then consider using sudo.

Running this in Linux will show just the pysc2 environment. If you run on either Mac or Windows, the game will also b run and you can see the game and the pysc2 environment.

## Dependencies

- pysc2 (Deepmind) [https://github.com/deepmind/pysc2]
- baselines (OpenAI) [https://github.com/openai/baselines]
- s2client-proto (Blizzard) [https://github.com/Blizzard/s2client-proto]
- Tensorflow 1.5 (Google) [https://github.com/tensorflow/tensorflow]

## Usage


## 1. Get Dependencies

### PyPI

The easiest way to get PySC2 is to use pip:

```shell
$ pip3 install pysc2
```

Before you install `baselines` you will require cmake.
```shell
$ sudo apt-get cmake
```

You will also need tensorflow. Tensorflow GPU only works with nVidia graphics due to implementation around CUDA cores.
CURRENTLY TRYING TO FIUGRE OUT HOW TO GET TENSORFLOW-GPU TO WORK.
```shell
$ pip3 install tensorflow
$ pip3 install tensorflow-gpu
```

Once you install tensorflow-gpu, it will run the GPU. To revert, you could:
1. Reinstall tensorflow
```shell
$ pip3 uninstall tensorflow
$ pip3 install tensorflow --ignore-install
```

Also, you have to install `baselines` library. This is where the basic ML algorithms are. It is not required to do MARL but is a good starting point.

```shell
$ pip3 install baselines
```

```shell
$ pip3 install absl-py
```

I don't think the following is necessary but may help with errors.
```shell
$ pip3 install python-gflags
```

## 2. Install StarCraft II

### Mac / Win

You have to download StarCraft II and install it. StarCraft II is now free.

http://us.battle.net/sc2/en/legacy-of-the-void/

### Linux Packages

Follow Blizzard's [documentation](https://github.com/Blizzard/s2client-proto#downloads) to
get the linux version. By default, PySC2 expects the game to live in
`~/StarCraftII/`.

* [4.0.2](http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.0.2.zip)

## 3. Download Maps

Download the [ladder maps](https://github.com/Blizzard/s2client-proto#downloads)
and the [mini games](https://github.com/deepmind/pysc2/releases/download/v1.0/mini_games.zip)
and extract them to your `StarcraftII/Maps/` directory.

## 4. Train it!

```shell
$ python3 train_mineral_shards.py
```

Once you hit a value that you are satisfied with then quit. I have found the the current one only goes up to about 14-15.

## 5. Enjoy it!

```shell
$ python3 enjoy_mineral_shards.py
```
