# COMP3096 - Research Methods
Thanks to [chris-chris](https://github.com/chris-chris/pysc2-examples) and [Siraj Raval](https://github.com/Zacharias030/A-Guide-to-DeepMinds-StarCraft-AI-Environment)

## Overview

This is the code for [this](https://youtu.be/URWXG5jRB-A) video on on Youtube by Siraj Raval. This code will help you train or run a pretrained AI model in the DeepMind Starcraft II environment.

The following doesn't work with Python 2. You will require Python 3. So run all commands with pip3 and python3. If there are any problems in the installation then consider using sudo.

Running this in Linux will show just the pysc2 environment. If you run on either Mac or Windows, the game will also be run and you can see the game and the pysc2 environment.

## Dependencies

- pysc2 (Deepmind) [https://github.com/deepmind/pysc2]
- baselines (OpenAI) [https://github.com/openai/baselines]
- s2client-proto (Blizzard) [https://github.com/Blizzard/s2client-proto]
- Tensorflow 1.5 (Google) [https://github.com/tensorflow/tensorflow]

This whole tensorflow, tensorflow-gpu and baselines are a dependencies nightmare.

## Usage


## 1. Get Dependencies

### PyPI

The easiest way to get PySC2 is to use pip:

```shell
$ sudo pip3 install pysc2
```

Before you install `baselines` you will require cmake.
```shell
$ sudo apt install cmake
```

You will also need tensorflow. Tensorflow GPU only works with nVidia graphics due to implementation around CUDA cores.
```shell
$ pip3 install tensorflow
$ pip3 install tensorflow-gpu
```

How to get tensorflow-gpu to work.
Install Nvidia driver above 380. If you have an existing driver then:
```shell
$ sudo apt-get purge nvidia*
```

To install: (It works for 390)
```shell
$ sudo add-apt-repository ppa:graphics-drivers
$ sudo apt-get update
$ sudo apt-get install nvidia-390
```

Reboot the computer and then check the driver with:
```shell
$ lsmod | grep nvidia
```

Next install the CUDA Toolkit 9.0 from [here](https://developer.nvidia.com/cuda-90-download-archive) using runfile (local). 9.1 isn't yet supported by TensorFlow 1.5
Then cd into the download directory and run this (Example is for x86-64 Ubuntu):
```shell
$ sudo sh cuda_9.0.176_384.81_linux.run
```
Don't install the Nvidia Driver and keep everything else as default.

Then add following line to bashrc:
```shell
$ nano ~/.bashrc

export LD_LIBRARY_PATH=LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64/
```

Now to install cuDNN. Download the install [here] (https://developer.nvidia.com/rdp/cudnn-download). You want the cuDNN v7.0.5 for CUDA 9.0.
Then do the following:
```shell
$ sudo rm -rf /usr/local/cuda/include/cudnn.h
$ sudo rm -rf /usr/local/cuda/lib64/libcudnn*
$ sudo tar xvzf cudnn-9.0-linux-x64-v7.tgz
$ cd cuda
$ sudo cp include/cudnn.h /usr/local/cuda/include/
$ sudo cp lib64/lib* /usr/local/cuda/lib64/
$ cd /usr/local/cuda/lib64/
$ sudo chmod +r libcudnn.so.7
$ sudo ldconfig
```

Tensorflow-GPU should now work. You can see GPU usage using:
```shell
watch -d -n 0.5 nvidia-smi
```

Once you install tensorflow-gpu, it will run the GPU. To revert, you could:
1. Reinstall tensorflow
```shell
$ pip3 uninstall tensorflow
$ pip3 install tensorflow --ignore-install
```

Also, you have to install `baselines` library. This is where the basic ML algorithms are. It is not required to do MARL but is a good starting point. The most recent version of `baselines` only runs on python3. So be warned. Although code for Raymond's and Sam's scripts should both run on python3. Refer to the last steps if you cannot run them.

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

https://starcraft2.com/en-gb/

### Linux Packages

Follow Blizzard's [documentation](https://github.com/Blizzard/s2client-proto#downloads) to
get the linux version. By default, PySC2 expects the game to live in
`~/StarCraftII/`.

* [4.0.2](http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.0.2.zip)

## 3. Download Maps

Download the [maps and melee packs](https://github.com/Blizzard/s2client-proto#downloads)
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

## 6. Using Sam's Script
Look at the command file in the experiments folder. Install the missing modules. It should just be sklearn. Remember to use pip3. You might have to use sudo pip3 install.

## 7. Using Raymond's Script
You have to pull Deepmind's pysc2 git folder. Then while inside the directory run the following:
```shell
$ sudo pip3 install -e .
```
After which you can then run the command in command file and it should work. If you don't run the command then when you attempt to run the command you will get an error saying PLAYER_RELATIVE cannot be resolved.
