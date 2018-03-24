#================================
# RESEARCH GROUP PROJECT [RGP]
#================================
# This file is from pekaalto/sc2atari.
# Source: https://github.com/pekaalto/sc2atari
# Important: Note that the action space (ac_space) parameter is never used and effectively *ignored*!

import tensorflow as tf
from pysc2.lib.features import SCREEN_FEATURES
from tensorflow.contrib import layers
from baselines.a2c.utils import sample
from baselines.common.distributions import make_pdtype
import math


class FullyConvPolicy:
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc * 1)
        X = tf.placeholder(tf.int32, ob_shape)  # obs

        with tf.variable_scope("fullyconv_model", reuse=reuse):
            x_onehot = layers.one_hot_encoding(
                # assuming we have only one channel
                X[:, :, :, 0],
                num_classes=SCREEN_FEATURES.player_relative.scale
            )

            #don't one hot 0-category
            x_onehot = x_onehot[:, :, :, 1:]

            h = layers.conv2d(
                x_onehot,
                num_outputs=16,
                kernel_size=5,
                stride=1,
                padding='SAME',
                scope="conv1"
            )
            h2 = layers.conv2d(
                h,
                num_outputs=32,
                kernel_size=3,
                stride=1,
                padding='SAME',
                scope="conv2"
            )
            pi = layers.flatten(layers.conv2d(
                h,
                num_outputs=int(math.sqrt(ac_space.n)), # RGP: Originally 1 (EXPERIMENTAL CHANGE)
                kernel_size=1,
                stride=1,
                scope="spatial_action",
                activation_fn=None
            ))

            pi *= 3.0  # make it little bit more deterministic, not sure if good idea

            f = layers.fully_connected(
                layers.flatten(h2),
                num_outputs=64,
                activation_fn=tf.nn.relu,
                scope="value_h_layer"
            )

            vf = layers.fully_connected(
                f,
                num_outputs=1,
                activation_fn=None,
                scope="value_out"
            )

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        v0 = vf[:, 0]
        a0 = sample(pi)
        # a0 = self.pd.sample()  # RGP (this is what other baselines policies do)
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None  # not stateful

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, v0, neglogp0], {X: ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X: ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
