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


def train():
    FLAGS(sys.argv)

    env = make_env(
        map_name=FLAGS.map_name,
        feature_screen_size=FLAGS.screen_size,
        feature_minimap_size=FLAGS.minimap_size,
        visualize=FLAGS.visualize,
    )

    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=True
    )

    act = deepq.learn(
        env,
        q_func=model,
        # lr=1e-5,
        # max_timesteps=10000000,
        # buffer_size=50000,
        # exploration_fraction=0.5,
        # exploration_final_eps=0.01,
        # train_freq=4,
        print_freq=10,
        learning_starts=1000,
        # target_network_update_freq=500,
        # gamma=1.0,
        # prioritized_replay=True,
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
