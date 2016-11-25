"""Trains the CelebA classifier."""
import argparse
import logging

import numpy
from blocks.algorithms import GradientDescent, Adam
from blocks.bricks import Rectifier, Logistic, MLP
from blocks.bricks.bn import SpatialBatchNormalization
from blocks.bricks.conv import Convolutional, ConvolutionalSequence
from blocks.bricks.cost import BinaryCrossEntropy
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.filter import VariableFilter
from blocks.graph import (ComputationGraph, apply_batch_normalization,
                          get_batch_normalization_updates, apply_dropout)
from blocks.initialization import IsotropicGaussian, Constant
from blocks.main_loop import MainLoop
from blocks.roles import OUTPUT
from theano import tensor
from blocks.model import Model

import cPickle as pkl

from celebA_fuel_loader import create_celeba_streams


def create_model_bricks():
    convnet = ConvolutionalSequence(
        layers=[
            Convolutional(
                filter_size=(4, 4),
                num_filters=32,
                name='conv1'),
            #SpatialBatchNormalization(name='batch_norm1'),
            Rectifier(),
            Convolutional(
                filter_size=(3, 3),
                step=(2, 2),
                num_filters=32,
                name='conv2'),
            #SpatialBatchNormalization(name='batch_norm2'),
            Rectifier(),
            Convolutional(
                filter_size=(4, 4),
                num_filters=64,
                name='conv3'),
            #SpatialBatchNormalization(name='batch_norm3'),
            Rectifier(),
            Convolutional(
                filter_size=(3, 3),
                step=(2, 2),
                num_filters=64,
                name='conv4'),
            #SpatialBatchNormalization(name='batch_norm4'),
            Rectifier(),
            Convolutional(
                filter_size=(3, 3),
                num_filters=128,
                name='conv5'),
            #SpatialBatchNormalization(name='batch_norm5'),
            Rectifier(),
            Convolutional(
                filter_size=(3, 3),
                step=(2, 2),
                num_filters=128,
                name='conv6'),
            #SpatialBatchNormalization(name='batch_norm6'),
            Rectifier(),
        ],
        num_channels=3,
        image_size=(128, 128),
        use_bias=True,
        weights_init=IsotropicGaussian(0.033),
        biases_init=Constant(0),
        name='convnet')
    convnet.initialize()

    mlp = MLP(
        activations=[Rectifier(), Logistic()],
        dims=[numpy.prod(convnet.get_dim('output')), 1000, 40],
        weights_init=IsotropicGaussian(0.033),
        biases_init=Constant(0),
        name='mlp')
    mlp.initialize()
    print(convnet)

    return convnet, mlp


def create_training_computation_graphs():
    x = tensor.tensor4('features')
    y = tensor.imatrix('targets')

    convnet, mlp = create_model_bricks()
    y_hat = mlp.apply(convnet.apply(x).flatten(ndim=2))
    cost = BinaryCrossEntropy().apply(y, y_hat)
    accuracy = 1 - tensor.neq(y > 0.5, y_hat > 0.5).mean()
    cg = ComputationGraph([cost, accuracy])

    # Create a graph which uses batch statistics for batch normalization
    # as well as dropout on selected variables
    #bn_cg = apply_batch_normalization(cg)
    #bricks_to_drop = ([convnet.layers[i] for i in (5, 11, 17)] +
    #                  [mlp.application_methods[1].brick])
    #variables_to_drop = VariableFilter(
    #    roles=[OUTPUT], bricks=bricks_to_drop)(cg.variables)
    #bn_dropout_cg = apply_dropout(cg, variables_to_drop, 0.5)

    return cg#, bn_dropout_cg


def run():
    streams = create_celeba_streams(training_batch_size=100,
                                    monitoring_batch_size=500,
                                    quality='128')
    main_loop_stream = streams[0]
    train_monitor_stream = streams[1]
    valid_monitor_stream = streams[2]

    cg = create_training_computation_graphs()

    model = Model(cg)

    # (params) = model.get_parameter_dict()
    #
    # f = open('params.pkl', 'w')
    #
    # pkl.dump(params, f)
    #
    # f.close()
    #
    # # print(params)
    # orz1 = params.popitem()
    # orz2 = params.popitem()
    # orz3 = params.popitem()
    #
    # orzorz = params.items()[0]
    # print(orzorz)
    #
    # print(orzorz[1])
    #
    # for attr in dir(orz2):
    #     print attr

    # numpy.save('orz', (orz1, orz2, orz3))


    # Compute parameter updates for the batch normalization population
    # statistics. They are updated following an exponential moving average.
    #pop_updates = get_batch_normalization_updates(bn_dropout_cg)
    decay_rate = 0.05
    #extra_updates = [(p, m * decay_rate + p * (1 - decay_rate))
    #                 for p, m in pop_updates]

    # Prepare algorithm
    step_rule = Adam()
    algorithm = GradientDescent(cost=cg.outputs[0],
                                parameters=cg.parameters,
                                step_rule=step_rule)
    #algorithm.add_updates(extra_updates)

    # Prepare monitoring
    cost = cg.outputs[0]
    cost.name = 'cost'
    train_monitoring = DataStreamMonitoring(
        [cost], train_monitor_stream, prefix="train",
        before_first_epoch=False, after_epoch=False, after_training=True)

    cost, accuracy = cg.outputs
    cost.name = 'cost'
    accuracy.name = 'accuracy'
    monitored_quantities = [cost, accuracy]
    valid_monitoring = DataStreamMonitoring(
        monitored_quantities, valid_monitor_stream, prefix="valid",
        before_first_epoch=False, after_epoch=False, every_n_epochs=5)

    # Prepare checkpoint
    checkpoint = Checkpoint(
        'celebA_128_50_noBN_bias.zip', every_n_epochs=10, use_cpickle=True)

    extensions = [Timing(), FinishAfter(after_n_epochs=50), train_monitoring,
                  valid_monitoring, checkpoint, Printing(), ProgressBar()]
    main_loop = MainLoop(data_stream=main_loop_stream, algorithm=algorithm,
                         extensions=extensions, model=model)
    main_loop.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Train a classifier on the CelebA dataset")
    args = parser.parse_args()
    run()
