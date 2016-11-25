from fuel.datasets import CelebA
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream

"""
    use fuel to load training sample.

"""


def create_streams(train_set, valid_set, test_set, training_batch_size,
                   monitoring_batch_size):
    """Creates data streams for training and monitoring.

    Parameters
    ----------
    train_set : :class:`fuel.datasets.Dataset`
        Training set.
    valid_set : :class:`fuel.datasets.Dataset`
        Validation set.
    test_set : :class:`fuel.datasets.Dataset`
        Test set.
    monitoring_batch_size : int
        Batch size for monitoring.
    include_targets : bool
        If ``True``, use both features and targets. If ``False``, use
        features only.

    Returns
    -------
    rval : tuple of data streams
        Data streams for the main loop, the training set monitor,
        the validation set monitor and the test set monitor.

    """
    main_loop_stream = DataStream.default_stream(
        dataset=train_set,
        iteration_scheme=ShuffledScheme(
            train_set.num_examples, training_batch_size))
    train_monitor_stream = DataStream.default_stream(
        dataset=train_set,
        iteration_scheme=ShuffledScheme(
            train_set.num_examples, monitoring_batch_size))
    valid_monitor_stream = DataStream.default_stream(
        dataset=valid_set,
        iteration_scheme=ShuffledScheme(
            valid_set.num_examples, monitoring_batch_size))
    test_monitor_stream = DataStream.default_stream(
        dataset=test_set,
        iteration_scheme=ShuffledScheme(
            test_set.num_examples, monitoring_batch_size))

    return (main_loop_stream, train_monitor_stream, valid_monitor_stream,
            test_monitor_stream)


def create_celeba_streams(training_batch_size=100, monitoring_batch_size=100, quality='128'):
    """Creates CelebA data streams.

    Parameters
    ----------
    training_batch_size : int
        Batch size for training.
    monitoring_batch_size : int
        Batch size for monitoring.
    include_targets : bool
        If ``True``, use both features and targets. If ``False``, use
        features only.

    Returns
    -------
    rval : tuple of data streams
        Data streams for the main loop, the training set monitor,
        the validation set monitor and the test set monitor.

    """
    sources = ('features', 'targets')

    train_set = CelebA(quality, ('train',), sources=sources)
    valid_set = CelebA(quality, ('valid',), sources=sources)
    test_set = CelebA(quality, ('test',), sources=sources)

    return create_streams(train_set, valid_set, test_set, training_batch_size,
                          monitoring_batch_size)

from blocks.serialization import load


"""
    in discgen's classifier, the class `checkpoint` call `secure_dump()`
    to save `mainloop`.
    to obtain the params, we must pass a `model` when initialize `mainloop`.
    and then utilize `model.get_parameter_values()` to get params.

    refer to:
    https://github.com/mila-udem/blocks/blob/master/blocks/extensions/saveload.py#L93
    https://github.com/mila-udem/blocks/blob/master/blocks/serialization.py#L238
"""


def load_encoder_params(file_name = 'celebA_100_new.zip'):
    f = open(file_name, 'rb')
    # print(f)
    main_loop = load(f, use_cpickle=True)
    print(main_loop.model)
    # refer to: https://blocks.readthedocs.io/en/latest/api/model.html#blocks.model.Model.get_parameter_values
    params_dict = main_loop.model.get_parameter_dict()
    params_name = []
    params = main_loop.model.get_parameter_values()

    for i in range(len(params_dict)):
        params_name.append(params_dict.items()[i][0])
        print(params_name[i])
        print(params[params_name[i]])

    return params_name, params_dict

# load_encoder_params()