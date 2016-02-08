"""Trains a VAE on the CIFAR10 dataset."""
import argparse
import logging

import numpy
import theano
from blocks.algorithms import GradientDescent, Adam
from blocks.bricks import Sequence, Random, Rectifier, Identity, MLP, Logistic
from blocks.bricks.bn import (BatchNormalization, BatchNormalizedMLP,
                              SpatialBatchNormalization)
from blocks.bricks.conv import (Convolutional, ConvolutionalTranspose,
                                ConvolutionalSequence)
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.extensions.saveload import Checkpoint
from blocks.filter import VariableFilter
from blocks.graph import (ComputationGraph, get_batch_normalization_updates,
                          apply_batch_normalization)
from blocks.initialization import IsotropicGaussian, Constant
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.roles import add_role, OUTPUT, PARAMETER
from blocks.select import Selector
from blocks.serialization import load
from blocks.utils import find_bricks, shared_floatx
from theano import tensor

from utils import create_cifar10_streams, load_vgg_classifier


def create_model_bricks():
    encoder_convnet = ConvolutionalSequence(
        layers=[
            Convolutional(
                filter_size=(5, 5),
                border_mode=(2, 2),
                num_filters=64,
                name='conv1'),
            Rectifier(),
            Convolutional(
                filter_size=(5, 5),
                border_mode=(2, 2),
                num_filters=64,
                name='conv2'),
            Rectifier(),
            Convolutional(
                filter_size=(5, 5),
                border_mode=(2, 2),
                step=(2, 2),
                num_filters=64,
                name='conv3'),
            Rectifier(),
            Convolutional(
                filter_size=(5, 5),
                border_mode=(2, 2),
                num_filters=128,
                name='conv4'),
            Rectifier(),
            Convolutional(
                filter_size=(5, 5),
                border_mode=(2, 2),
                num_filters=128,
                name='conv5'),
            Rectifier(),
            Convolutional(
                filter_size=(5, 5),
                border_mode=(2, 2),
                step=(2, 2),
                num_filters=128,
                name='conv6'),
            Rectifier(),
            Convolutional(
                filter_size=(5, 5),
                border_mode=(2, 2),
                num_filters=256,
                name='conv7'),
            Rectifier(),
            Convolutional(
                filter_size=(5, 5),
                border_mode=(2, 2),
                num_filters=256,
                name='conv8'),
            Rectifier(),
            Convolutional(
                filter_size=(5, 5),
                border_mode=(2, 2),
                step=(2, 2),
                num_filters=512,
                name='conv9'),
            Rectifier(),
        ],
        num_channels=3,
        image_size=(32, 32),
        tied_biases=True,
        weights_init=IsotropicGaussian(0.001),
        biases_init=Constant(0),
        name='encoder_convnet')
    encoder_convnet.initialize()

    encoder_filters = numpy.prod(encoder_convnet.get_dim('output'))

    encoder_mlp = MLP(
        dims=[encoder_filters, 4096, 4096, 4096, 4096, 4096],
        activations=[Sequence([BatchNormalization(4096).apply,
                               Rectifier().apply], name='activation1'),
                     Sequence([BatchNormalization(4096).apply,
                               Rectifier().apply], name='activation2'),
                     Sequence([BatchNormalization(4096).apply,
                               Rectifier().apply], name='activation3'),
                     Sequence([BatchNormalization(4096).apply,
                               Rectifier().apply], name='activation4'),
                     Identity().apply],
        weights_init=IsotropicGaussian(0.001),
        biases_init=Constant(0),
        use_bias=False,
        name='encoder_mlp')
    encoder_mlp.push_allocation_config()
    encoder_mlp.linear_transformations[-1].use_bias = True
    encoder_mlp.initialize()

    decoder_mlp = BatchNormalizedMLP(
        activations=[Rectifier(), Rectifier()],
        dims=[encoder_mlp.output_dim // 2, 4096, encoder_filters],
        weights_init=IsotropicGaussian(0.033),
        biases_init=Constant(0),
        use_bias=False,
        name='decoder_mlp')
    decoder_mlp.initialize()

    decoder_convnet = ConvolutionalSequence(
        layers=[
            Convolutional(
                filter_size=(1, 1),
                num_filters=256,
                name='conv1'),
            Rectifier(),
            Convolutional(
                filter_size=(1, 1),
                num_filters=256,
                name='conv2'),
            Rectifier(),
            ConvolutionalTranspose(
                filter_size=(2, 2),
                step=(2, 2),
                original_image_size=(8, 8),
                num_filters=256,
                name='conv3'),
            Rectifier(),
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=256,
                name='conv4'),
            Rectifier(),
            Convolutional(
                filter_size=(3, 3),
                border_mode=(1, 1),
                num_filters=128,
                name='conv5'),
            Rectifier(),
            ConvolutionalTranspose(
                filter_size=(2, 2),
                step=(2, 2),
                original_image_size=(16, 16),
                num_filters=128,
                name='conv6'),
            Rectifier(),
            Convolutional(
                filter_size=(5, 5),
                border_mode=(2, 2),
                num_filters=128,
                name='conv7'),
            Rectifier(),
            Convolutional(
                filter_size=(5, 5),
                border_mode=(2, 2),
                num_filters=64,
                name='conv8'),
            Rectifier(),
            ConvolutionalTranspose(
                filter_size=(2, 2),
                step=(2, 2),
                original_image_size=(32, 32),
                num_filters=64,
                name='conv9'),
            Rectifier(),
            Convolutional(
                filter_size=(5, 5),
                border_mode=(2, 2),
                num_filters=64,
                name='conv10'),
            Rectifier(),
            Convolutional(
                filter_size=(5, 5),
                border_mode=(2, 2),
                num_filters=64,
                name='conv11'),
            Rectifier(),
            Convolutional(
                filter_size=(5, 5),
                border_mode=(2, 2),
                num_filters=3,
                name='conv12'),
            Logistic(),
        ],
        num_channels=encoder_convnet.get_dim('output')[0],
        image_size=encoder_convnet.get_dim('output')[1:],
        tied_biases=True,
        weights_init=IsotropicGaussian(0.033),
        biases_init=Constant(0),
        name='decoder_convnet')
    decoder_convnet.initialize()

    return encoder_convnet, encoder_mlp, decoder_convnet, decoder_mlp


def create_training_computation_graphs(discriminative_regularization):
    x = tensor.tensor4('features')
    pi = numpy.cast[theano.config.floatX](numpy.pi)

    bricks = create_model_bricks()
    encoder_convnet, encoder_mlp, decoder_convnet, decoder_mlp = bricks
    if discriminative_regularization:
        classifier_convnet = load_vgg_classifier()
        conv_indices = [0, 5, 10, 12, 14, 19, 28, 34]
    random_brick = Random()

    # Initialize conditional variances
    log_sigma_theta = shared_floatx(
        numpy.zeros((3, 32, 32)), name='log_sigma_theta')
    add_role(log_sigma_theta, PARAMETER)
    variance_parameters = [log_sigma_theta]
    if discriminative_regularization:
        # We add discriminative regularization for the batch-normalized output
        # of the strided layers of the classifier.
        for i in conv_indices:
            layer = classifier_convnet.layers[i]
            log_sigma = shared_floatx(
                numpy.zeros(layer.get_dim('output')),
                name='{}_log_sigma'.format(layer.name))
            add_role(log_sigma, PARAMETER)
            variance_parameters.append(log_sigma)

    # Encode
    phi = encoder_mlp.apply(encoder_convnet.apply(x).flatten(ndim=2))
    nlat = encoder_mlp.output_dim // 2
    mu_phi = phi[:, :nlat]
    log_sigma_phi = phi[:, nlat:]
    # Sample from the approximate posterior
    epsilon = random_brick.theano_rng.normal(
        size=mu_phi.shape, dtype=mu_phi.dtype)
    z = mu_phi + epsilon * tensor.exp(log_sigma_phi)
    # Decode
    mu_theta = decoder_convnet.apply(
        decoder_mlp.apply(z).reshape(
            (-1,) + decoder_convnet.get_dim('input_')))
    log_sigma = log_sigma_theta.dimshuffle('x', 0, 1, 2)

    # Compute KL and reconstruction terms
    kl_term = 0.5 * (
        tensor.exp(2 * log_sigma_phi) + mu_phi ** 2 - 2 * log_sigma_phi - 1
    ).sum(axis=1)
    reconstruction_term = -0.5 * (
        tensor.log(2 * pi) + 2 * log_sigma +
        (x - mu_theta) ** 2 / tensor.exp(2 * log_sigma)
    ).sum(axis=[1, 2, 3])
    total_reconstruction_term = reconstruction_term

    if discriminative_regularization:
        # Propagate both the input and the reconstruction through the
        # classifier
        acts_cg = ComputationGraph(
            [classifier_convnet.apply(255 * x[:, ::-1, :, :])])
        acts_hat_cg = ComputationGraph(
            [classifier_convnet.apply(255 * mu_theta[:, ::-1, :, :])])

        # Retrieve activations of interest and compute discriminative
        # regularization reconstruction terms
        for i, log_sigma in zip(conv_indices, variance_parameters[1:]):
            layer = classifier_convnet.layers[i]
            variable_filter = VariableFilter(roles=[OUTPUT],
                                             bricks=[layer])
            d, = variable_filter(acts_cg)
            d_hat, = variable_filter(acts_hat_cg)
            log_sigma = log_sigma.dimshuffle('x', 0, 1, 2)

            total_reconstruction_term += -0.5 * (
                tensor.log(2 * pi) + 2 * log_sigma +
                (d - d_hat) ** 2 / tensor.exp(2 * log_sigma)
            ).sum(axis=[1, 2, 3])

    cost = (kl_term - total_reconstruction_term).mean()

    cg = ComputationGraph([cost, kl_term, reconstruction_term])

    return cg, apply_batch_normalization(bn_cg), variance_parameters


def run(discriminative_regularization=True):
    streams = create_celeba_streams(training_batch_size=100,
                                    monitoring_batch_size=500,
                                    include_targets=False)
    main_loop_stream, train_monitor_stream, valid_monitor_stream = streams[:3]

    # Compute parameter updates for the batch normalization population
    # statistics. They are updated following an exponential moving average.
    rval = create_training_computation_graphs(discriminative_regularization)
    cg, bn_cg, variance_parameters = rval
    pop_updates = list(
        set(get_batch_normalization_updates(bn_cg, allow_duplicates=True)))
    decay_rate = 0.05
    extra_updates = [(p, m * decay_rate + p * (1 - decay_rate))
                     for p, m in pop_updates]

    model = Model(bn_cg.outputs[0])
    selector = Selector(
        find_bricks(
            model.top_bricks,
            lambda brick: brick.name in ('encoder_convnet', 'encoder_mlp',
                                         'decoder_convnet', 'decoder_mlp')))
    parameters = list(selector.get_parameters().values()) + variance_parameters

    # Prepare algorithm
    step_rule = Adam()
    algorithm = GradientDescent(cost=bn_cg.outputs[0],
                                parameters=parameters,
                                step_rule=step_rule)
    algorithm.add_updates(extra_updates)

    # Prepare monitoring
    monitored_quantities_list = []
    for graph in [bn_cg, cg]:
        cost, kl_term, reconstruction_term = graph.outputs
        cost.name = 'nll_upper_bound'
        avg_kl_term = kl_term.mean(axis=0)
        avg_kl_term.name = 'avg_kl_term'
        avg_reconstruction_term = -reconstruction_term.mean(axis=0)
        avg_reconstruction_term.name = 'avg_reconstruction_term'
        monitored_quantities_list.append(
            [cost, avg_kl_term, avg_reconstruction_term])
    train_monitoring = DataStreamMonitoring(
        monitored_quantities_list[0], train_monitor_stream, prefix="train",
        updates=extra_updates, every_n_epochs=5)
    valid_monitoring = DataStreamMonitoring(
        monitored_quantities_list[1], valid_monitor_stream, prefix="valid",
        every_n_epochs=5)

    # Prepare checkpoint
    save_path = 'cifar_vae_{}regularization.zip'.format(
        '' if discriminative_regularization else 'no_')
    checkpoint = Checkpoint(save_path, every_n_epochs=5, use_cpickle=True)

    extensions = [Timing(), FinishAfter(after_n_epochs=75), train_monitoring,
                  valid_monitoring, checkpoint, Printing(), ProgressBar()]
    main_loop = MainLoop(data_stream=main_loop_stream,
                         algorithm=algorithm, extensions=extensions)
    main_loop.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Train a VAE on the CIFAR10 dataset")
    parser.add_argument("--regularize", action='store_true',
                        help="apply discriminative regularization")
    args = parser.parse_args()
    run(args.regularize)
