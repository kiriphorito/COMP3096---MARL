#================================
# RESEARCH GROUP PROJECT [RGP]
#================================
# This file is part of the COMP3096 Research Group Project.

# ===============
# Imports
# ===============
# System / Settings
import sys
from absl import flags
from absl.flags import FLAGS
# Environment
from sc2g.env.sc2gym import make_env
# Algorithm
from baselines import deepq
from baselines.common.atari_wrappers import FrameStack


def train():
    FLAGS(sys.argv)

    env = make_env(
        map_name=FLAGS.map_name,
        feature_screen_size=FLAGS.screen_size,
        feature_minimap_size=FLAGS.minimap_size,
        visualize=FLAGS.visualize,
    )

    env = FrameStack(env, 4)

    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=True
    )

    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-5,                         # 5e-4: learning rate for adam optimizer
        max_timesteps=5000000,           # 100000
        buffer_size=100000,              # 50000
        exploration_fraction=0.5,        # 0.1
        exploration_final_eps=0.01,      # 0.02
        train_freq=4,                    # 1: how often the model is updated, in steps
        print_freq=10,                   # 100: how often training progress is printed, in episodes
        checkpoint_freq=10000,           # 10000: how often to save the model, in steps
        learning_starts=100000,          # 1000: how many steps before learning starts
        gamma=0.99,                      # 1.0: discount factor
        target_network_update_freq=500,  # 500: how often the target network is updated
        prioritized_replay=True,
    )

    try:
        print("Saving model to {}_model.pkl".format(FLAGS.map_name))
        act.save("{}_model.pkl".format(FLAGS.map_name))
    except:
        print("Error saving model.")

    print("Saving replay...")
    env.save_replay(FLAGS.map_name)

    print("Closing environment...")
    env.close()


def main():
    flags.DEFINE_string("map_name", "CollectMineralShards", "Name of the map")
    flags.DEFINE_integer("screen_size", 84, "Feature screen size")
    flags.DEFINE_integer("minimap_size", 64, "Feature minimap size")
    flags.DEFINE_bool("visualize", False, "Show python visualisation")

    train()


if __name__ == "__main__":
    main()
