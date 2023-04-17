from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.mlp.fully_conn import *
from lib.mlp.layer_utils import *
from lib.cnn.layer_utils import *


""" Classes """
class TestCNN(Module):
    def __init__(self, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            ConvLayer2D(input_channels=3, kernel_size=3, number_filters=3, stride=1, padding=0, name='convolution1'),
            MaxPoolingLayer(2, 2, 'pooling'),
            flatten("flat"),
            fc(27, 5, 0.02, 'fully1')
            ########### END ###########
        )


class SmallConvolutionalNetwork(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            ConvLayer2D(input_channels=3, kernel_size=3, number_filters=20, stride=1, padding=1, name='convolution1'),
            gelu(name='g1'),
            ConvLayer2D(input_channels=20, kernel_size=3, number_filters=40, stride=2, padding=0, name='convolution2'),
            gelu(name='g2'),
            MaxPoolingLayer(2, 2, 'pooling1'),
            flatten("flat"),
            fc(1960, 600, 0.02, 'fully1'),
            gelu(name='g3'),
            fc(600, 20, 0.02, 'fully2')
            ########### END ###########
        )