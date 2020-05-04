import keras
import localmodule
import annotations
import keras.backend as K
import numpy as np


def create_classifier_base(n_input_hops, n_filters, n_hidden_units, kernel_size, pool_size, base_wd=0.001):
    # Input
    n_mels = localmodule.get_pcen_settings()["top_freq_id"]
    tfr_input = keras.layers.Input(shape=(n_mels, n_input_hops, 1), name="tfr_input")

    # Layer 1
    bn = keras.layers.normalization.BatchNormalization()(tfr_input)
    conv1 = keras.layers.Convolution2D(n_filters[0], kernel_size,
                                       padding="same", kernel_initializer="he_normal", activation="relu",
                                       name="conv1")(bn)
    pool1 = keras.layers.MaxPooling2D(pool_size=pool_size, name="pool1")(conv1)

    # Layer 2
    conv2 = keras.layers.Convolution2D(n_filters[1], kernel_size,
                                       padding="same", kernel_initializer="he_normal", activation="relu",
                                       name="conv2")(pool1)
    pool2 = keras.layers.MaxPooling2D(pool_size=pool_size, name="pool2")(conv2)

    # Layer 3
    conv3 = keras.layers.Convolution2D(n_filters[2], kernel_size,
                                       padding="same", kernel_initializer="he_normal", activation="relu",
                                       name="conv3")(pool2)

    # Layer 4
    flatten = keras.layers.Flatten()(conv3)
    dense1 = keras.layers.Dense(n_hidden_units,
                                kernel_initializer="he_normal", activation="relu",
                                kernel_regularizer=keras.regularizers.l2(base_wd),
                                use_bias=False,
                                name="dense1")(flatten)

    return tfr_input, dense1


def create_flat_singletask_fine_model(n_input_hops, n_filters, n_hidden_units, kernel_size, pool_size, base_wd=0.001):
    tfr_input, dense1 \
        = create_classifier_base(n_input_hops, n_filters, n_hidden_units, kernel_size, pool_size)

    wd = base_wd / (43.0 / annotations.NUM_MOD_FINE)
    dense_fine = keras.layers.Dense(annotations.NUM_MOD_FINE,
                                    kernel_initializer="normal", activation="softmax",
                                    kernel_regularizer=keras.regularizers.l2(wd),
                                    use_bias=False,
                                    name="y_fine")(dense1)

    return [tfr_input], [dense_fine]


def create_flat_singletask_medium_model(n_input_hops, n_filters, n_hidden_units, kernel_size, pool_size, base_wd=0.001):
    tfr_input, dense1 \
        = create_classifier_base(n_input_hops, n_filters, n_hidden_units, kernel_size, pool_size)

    wd = base_wd / (43.0 / annotations.NUM_MOD_MEDIUM)
    dense_medium = keras.layers.Dense(annotations.NUM_MOD_MEDIUM,
                                    kernel_initializer="normal", activation="softmax",
                                    kernel_regularizer=keras.regularizers.l2(wd),
                                    use_bias=False,
                                    name="y_medium")(dense1)

    return [tfr_input], [dense_medium]


def create_flat_singletask_coarse_model(n_input_hops, n_filters, n_hidden_units, kernel_size, pool_size, base_wd=0.001):
    tfr_input, dense1 \
        = create_classifier_base(n_input_hops, n_filters, n_hidden_units, kernel_size, pool_size)

    wd = base_wd / 43.0
    dense_coarse = keras.layers.Dense(1,
                                      kernel_initializer="normal", activation="sigmoid",
                                      kernel_regularizer=keras.regularizers.l2(wd),
                                      use_bias=False,
                                      name="y_coarse")(dense1)


    return [tfr_input], [dense_coarse]


# LAMBDA FUNCTIONS
def hierarchical_containment_layer_sum_activation(x):
    return K.tanh(K.sum(x, axis=-1, keepdims=True))


def other_activation(x):
    return 1.0 - K.max(x, axis=-1, keepdims=True)


def make_slice_func(start, end):
    def slice_func(x):
        return x[:,start:end]
    return slice_func


def create_hierarchical_containment_model(n_input_hops, n_filters, n_hidden_units, kernel_size, pool_size, base_wd):
    # Input
    n_mels = localmodule.get_pcen_settings()["top_freq_id"]
    tfr_input = keras.layers.Input(shape=(n_mels, n_input_hops, 1), name="tfr_input")

    # Layer 1
    bn = keras.layers.normalization.BatchNormalization()(tfr_input)
    conv1 = keras.layers.Convolution2D(n_filters[0], kernel_size,
                                       padding="same", kernel_initializer="he_normal", activation="relu",
                                       name="conv1")(bn)

    pool1 = keras.layers.MaxPooling2D(pool_size=pool_size, name="pool1")(conv1)

    # Layer 2
    conv2 = keras.layers.Convolution2D(n_filters[1], kernel_size,
                                       padding="same", kernel_initializer="he_normal", activation="relu",
                                       name="conv2")(pool1)
    pool2 = keras.layers.MaxPooling2D(pool_size=pool_size, name="pool2")(conv2)

    # Layer 3
    conv3 = keras.layers.Convolution2D(n_filters[2], kernel_size,
                                       padding="same", kernel_initializer="he_normal", activation="relu",
                                       name="conv3")(pool2)

    # Layer 4
    coarse_layer = keras.layers.Flatten()(conv3)

    y_coarse = keras.layers.Lambda(hierarchical_containment_layer_sum_activation, name="y_coarse")(coarse_layer)

    # Don't include other
    num_medium_classes = len(annotations.MOD_MEDIUM_COUNTS)
    medium_children_counts = [0 for _ in range(num_medium_classes)]
    for medium_code, medium_count in annotations.MOD_MEDIUM_COUNTS.items():
        medium_idx = annotations.MOD_MEDIUM_IDXS[medium_code]
        medium_children_counts[medium_idx] = medium_count - 1


    med_props = np.array(medium_children_counts) / float(sum(medium_children_counts))
    class_sublayer_sizes = partition(n_hidden_units, med_props)

    medium_layer = keras.layers.Dense(n_hidden_units,
                                kernel_initializer="he_normal", activation="relu", use_bias=False,
                                kernel_regularizer=keras.regularizers.l2(base_wd))(coarse_layer)

    medium_nodes = []
    fine_sublayers = []
    start = 0
    for med_idx, (sublayer_size, fine_count) in enumerate(zip(class_sublayer_sizes, medium_children_counts)):
        medium_sublayer = keras.layers.Lambda(make_slice_func(int(start), int(start+sublayer_size)), name="medium_sublayer_{}".format(med_idx))(medium_layer)

        #
        med_node = keras.layers.Lambda(hierarchical_containment_layer_sum_activation)(medium_sublayer)
        medium_nodes.append(med_node)

        start += sublayer_size

        # Make fine layer
        wd = base_wd / (43.0 / fine_count)
        fine_sublayer = keras.layers.Dense(fine_count, kernel_initializer="he_normal", activation="relu",
                                           use_bias=False, kernel_regularizer=keras.regularizers.l2(wd),
                                           name="fine_sublayer_{}".format(med_idx))(medium_sublayer)

        fine_sublayers.append(fine_sublayer)

    medium_output = keras.layers.Concatenate()(medium_nodes)

    fine_layer = keras.layers.Concatenate()(fine_sublayers)
    fine_output = keras.layers.Lambda(K.tanh)(fine_layer)

    medium_other_output = keras.layers.Lambda(other_activation)(medium_output)
    fine_other_output = keras.layers.Lambda(other_activation)(fine_output)

    y_medium = keras.layers.Concatenate(name="y_medium")([medium_output, medium_other_output])
    y_fine = keras.layers.Concatenate(name="y_fine")([fine_output, fine_other_output])

    return [tfr_input], [y_coarse, y_medium, y_fine]


def partition(n, prop_list):
    prop_list = np.array(prop_list, dtype='float32')
    ideal_part = n * prop_list
    round_part = ideal_part.astype(int)

    missing = n - round_part.sum()

    # To make sure we have the total count accounted for, add one to
    # elements that are most underneath the true proportion
    for _ in range(missing):
        idx = (ideal_part - round_part).argmax()
        round_part[idx] += 1

    assert n == round_part.sum()

    return list(round_part)


def partition_layer_output(layer, counts, n_units):
    props = np.array(counts) / float(np.sum(counts))
    if n_units is None:
        try:
            n_units = int(layer.shape[-1]) # Maybe this will work?
        except:
            import pdb
            pdb.set_trace()
    sublayer_sizes = partition(n_units, props)

    sublayer_list = []
    start = 0
    for sub_idx, sublayer_size in enumerate(sublayer_sizes):
        sublayer = keras.layers.Lambda(make_slice_func(int(start), int(start+sublayer_size)), name="partition_sublayer_{}".format(sub_idx))(layer)
        sublayer_list.append(sublayer)
        start += sublayer_size

    return sublayer_list


def create_nonhierarchical_multitask_model(n_input_hops, n_filters, n_hidden_units, kernel_size, pool_size, base_wd=0.001, name_outputs=True):
    tfr_input, dense1 \
        = create_classifier_base(n_input_hops, n_filters, n_hidden_units, kernel_size, pool_size)

    if name_outputs:
        medium_name = "y_medium"
        fine_name = "y_fine"
    else:
        medium_name = None
        fine_name = None

    wd = base_wd / 43.0
    y_coarse = keras.layers.Dense(1,
                                  kernel_initializer="normal", activation="sigmoid",
                                  kernel_regularizer=keras.regularizers.l2(wd),
                                  use_bias=False,
                                  name="y_coarse")(dense1)

    wd = base_wd / (43.0 / annotations.NUM_MOD_MEDIUM)
    y_medium = keras.layers.Dense(annotations.NUM_MOD_MEDIUM,
                                    kernel_initializer="normal", activation="softmax",
                                    kernel_regularizer=keras.regularizers.l2(wd),
                                    use_bias=False,
                                    name=medium_name)(dense1)

    wd = base_wd / (43.0 / annotations.NUM_MOD_FINE)
    y_fine = keras.layers.Dense(annotations.NUM_MOD_FINE,
                                    kernel_initializer="normal", activation="softmax",
                                    kernel_regularizer=keras.regularizers.l2(wd),
                                    use_bias=False,
                                    name=fine_name)(dense1)

    return [tfr_input], [y_coarse, y_medium, y_fine]


def create_taxonet_model(n_input_hops, n_filters, n_hidden_units, kernel_size, pool_size, base_wd):
    # Input
    n_mels = localmodule.get_pcen_settings()["top_freq_id"]
    tfr_input = keras.layers.Input(shape=(n_mels, n_input_hops, 1), name="tfr_input")

    # Layer 1
    bn = keras.layers.normalization.BatchNormalization()(tfr_input)
    conv1 = keras.layers.Convolution2D(n_filters[0], kernel_size,
                                       padding="same", kernel_initializer="he_normal", activation="relu",
                                       name="conv1")(bn)

    pool1 = keras.layers.MaxPooling2D(pool_size=pool_size, name="pool1")(conv1)

    # Layer 2
    conv2 = keras.layers.Convolution2D(n_filters[1], kernel_size,
                                       padding="same", kernel_initializer="he_normal", activation="relu",
                                       name="conv2")(pool1)

    pool2 = keras.layers.MaxPooling2D(pool_size=pool_size, name="pool2")(conv2)

    # Layer 3
    conv3 = keras.layers.Convolution2D(n_filters[2], kernel_size,
                                       padding="same", kernel_initializer="he_normal", activation="relu",
                                       name="conv3")(pool2)

    # Layer 4
    coarse_layer = keras.layers.Flatten()(conv3)

    wd = base_wd / 43.0
    y_coarse = keras.layers.Dense(1,
                                  kernel_initializer="normal", activation="sigmoid",
                                  kernel_regularizer=keras.regularizers.l2(wd),
                                  use_bias=False,
                                  name="y_coarse")(coarse_layer)

    # Don't include other
    num_medium_classes = len(annotations.MOD_MEDIUM_COUNTS)
    medium_children_counts = [0 for _ in range(num_medium_classes)]
    for medium_code, medium_count in annotations.MOD_MEDIUM_COUNTS.items():
        medium_idx = annotations.MOD_MEDIUM_IDXS[medium_code]
        medium_children_counts[medium_idx] = int(medium_count - 1)

    med_props = np.array(medium_children_counts) / float(sum(medium_children_counts))
    class_sublayer_sizes = partition(n_hidden_units, med_props)

    # Adjust number of hidden units so that we nicely partition layer in terms of medium classes
    n_hidden_units = sum(class_sublayer_sizes)

    medium_layer = keras.layers.Dense(n_hidden_units,
                                      kernel_initializer="he_normal", activation="relu", use_bias=False,
                                      kernel_regularizer=keras.regularizers.l2(base_wd))(coarse_layer)

    medium_nodes = []
    fine_sublayers = []
    start = 0
    for med_idx, (sublayer_size, fine_count) in enumerate(zip(class_sublayer_sizes, medium_children_counts)):
        medium_sublayer = keras.layers.Lambda(make_slice_func(int(start), int(start+sublayer_size)), name="medium_sublayer_{}".format(med_idx))(medium_layer)

        wd = base_wd / 43.0
        med_node = keras.layers.Dense(1,
                                      kernel_initializer="normal", activation="sigmoid",
                                      kernel_regularizer=keras.regularizers.l2(wd),
                                      use_bias=False)(medium_sublayer)

        medium_nodes.append(med_node)

        start += sublayer_size

        # Make fine layer
        wd = base_wd / (43.0 / fine_count)
        fine_sublayer = keras.layers.Dense(fine_count, kernel_initializer="he_normal", activation="sigmoid",
                                           use_bias=False, kernel_regularizer=keras.regularizers.l2(wd),
                                           name="fine_sublayer_{}".format(med_idx))(medium_sublayer)

        fine_sublayers.append(fine_sublayer)

    medium_output = keras.layers.Concatenate()(medium_nodes)

    fine_output = keras.layers.Concatenate()(fine_sublayers)

    medium_other_output = keras.layers.Lambda(other_activation)(medium_output)
    fine_other_output = keras.layers.Lambda(other_activation)(fine_output)

    y_medium = keras.layers.Concatenate(name="y_medium")([medium_output, medium_other_output])
    y_fine = keras.layers.Concatenate(name="y_fine")([fine_output, fine_other_output])

    return [tfr_input], [y_coarse, y_medium, y_fine]


def create_hierarchical_composition_multiclass_model(n_input_hops, n_filters, n_hidden_units, kernel_size, pool_size, base_wd):
    # Input
    n_mels = localmodule.get_pcen_settings()["top_freq_id"]
    tfr_input = keras.layers.Input(shape=(n_mels, n_input_hops, 1), name="tfr_input")

    # Layer 1
    bn = keras.layers.normalization.BatchNormalization()(tfr_input)
    conv1 = keras.layers.Convolution2D(n_filters[0], kernel_size,
                                       padding="same", kernel_initializer="he_normal", activation="relu",
                                       name="conv1")(bn)

    pool1 = keras.layers.MaxPooling2D(pool_size=pool_size, name="pool1")(conv1)

    # Layer 2
    conv2 = keras.layers.Convolution2D(n_filters[1], kernel_size,
                                       padding="same", kernel_initializer="he_normal", activation="relu",
                                       name="conv2")(pool1)

    pool2 = keras.layers.MaxPooling2D(pool_size=pool_size, name="pool2")(conv2)

    # Layer 3
    conv3 = keras.layers.Convolution2D(n_filters[2], kernel_size,
                                       padding="same", kernel_initializer="he_normal", activation="relu",
                                       name="conv3")(pool2)

    # Layer 4
    coarse_layer = keras.layers.Flatten()(conv3)

    wd = base_wd / 43.0
    y_coarse = keras.layers.Dense(1,
                                  kernel_initializer="normal", activation="sigmoid",
                                  kernel_regularizer=keras.regularizers.l2(wd),
                                  use_bias=False,
                                  name="y_coarse")(coarse_layer)

    # Don't include other
    num_medium_classes = len(annotations.MOD_MEDIUM_COUNTS)
    medium_children_counts = [0 for _ in range(num_medium_classes)]
    for medium_code, medium_count in annotations.MOD_MEDIUM_COUNTS.items():
        medium_idx = annotations.MOD_MEDIUM_IDXS[medium_code]
        medium_children_counts[medium_idx] = int(medium_count - 1)

    med_props = np.array(medium_children_counts) / float(sum(medium_children_counts))
    class_sublayer_sizes = partition(n_hidden_units, med_props)

    # Adjust number of hidden units so that we nicely partition layer in terms of medium classes
    n_hidden_units = sum(class_sublayer_sizes)

    medium_layer = keras.layers.Dense(n_hidden_units,
                                      kernel_initializer="he_normal", activation="relu", use_bias=False,
                                      kernel_regularizer=keras.regularizers.l2(base_wd))(coarse_layer)

    medium_nodes = []
    fine_sublayers = []
    start = 0
    for med_idx, (sublayer_size, fine_count) in enumerate(zip(class_sublayer_sizes, medium_children_counts)):
        medium_sublayer = keras.layers.Lambda(make_slice_func(int(start), int(start+sublayer_size)), name="medium_sublayer_{}".format(med_idx))(medium_layer)

        wd = base_wd / 43.0
        med_node = keras.layers.Dense(1,
                                      kernel_initializer="normal", activation="linear",
                                      kernel_regularizer=keras.regularizers.l2(wd),
                                      use_bias=False)(medium_sublayer)

        medium_nodes.append(med_node)

        start += sublayer_size

        # Make fine layer
        wd = base_wd / (43.0 / fine_count)
        fine_sublayer = keras.layers.Dense(fine_count, kernel_initializer="he_normal", activation="linear",
                                           use_bias=False, kernel_regularizer=keras.regularizers.l2(wd),
                                           name="fine_sublayer_{}".format(med_idx))(medium_sublayer)

        fine_sublayers.append(fine_sublayer)

    wd = base_wd / 43.0
    medium_other_node = keras.layers.Dense(1,
                                           kernel_initializer="normal", activation="linear",
                                           kernel_regularizer=keras.regularizers.l2(wd),
                                           use_bias=False)(medium_layer)
    medium_nodes.append(medium_other_node)
    medium_output = keras.layers.Concatenate()(medium_nodes)
    y_medium = keras.layers.Softmax(name="y_medium")(medium_output)

    fine_layer = keras.layers.Concatenate()(fine_sublayers)
    fine_other_node = keras.layers.Dense(1,
                                         kernel_initializer="normal", activation="linear",
                                         kernel_regularizer=keras.regularizers.l2(wd),
                                         use_bias=False)(fine_layer)
    fine_sublayers.append(fine_other_node)
    fine_output = keras.layers.Concatenate()(fine_sublayers)
    y_fine = keras.layers.Softmax(name="y_fine")(fine_output)

    return [tfr_input], [y_coarse, y_medium, y_fine]


def create_hierarchical_baseline_model(n_input_hops, n_filters, n_hidden_units, kernel_size, pool_size, base_wd):
    # Input
    n_mels = localmodule.get_pcen_settings()["top_freq_id"]
    tfr_input = keras.layers.Input(shape=(n_mels, n_input_hops, 1), name="tfr_input")

    # Layer 1
    bn = keras.layers.normalization.BatchNormalization()(tfr_input)
    conv1 = keras.layers.Convolution2D(n_filters[0], kernel_size,
                                       padding="same", kernel_initializer="he_normal", activation="relu",
                                       name="conv1")(bn)

    pool1 = keras.layers.MaxPooling2D(pool_size=pool_size, name="pool1")(conv1)

    # Layer 2
    conv2 = keras.layers.Convolution2D(n_filters[1], kernel_size,
                                       padding="same", kernel_initializer="he_normal", activation="relu",
                                       name="conv2")(pool1)

    pool2 = keras.layers.MaxPooling2D(pool_size=pool_size, name="pool2")(conv2)

    # Layer 3
    conv3 = keras.layers.Convolution2D(n_filters[2], kernel_size,
                                       padding="same", kernel_initializer="he_normal", activation="relu",
                                       name="conv3")(pool2)

    # Layer 4
    flattened = keras.layers.Flatten()(conv3)

    wd = base_wd / 43.0
    y_coarse = keras.layers.Dense(annotations.NUM_MOD_COARSE - 1,
                                  kernel_initializer="normal", activation="sigmoid",
                                  kernel_regularizer=keras.regularizers.l2(wd),
                                  use_bias=False,
                                  name="y_coarse")(flattened)


    dense1 = keras.layers.Dense(n_hidden_units,
                                      kernel_initializer="he_normal", activation="relu", use_bias=False,
                                      kernel_regularizer=keras.regularizers.l2(base_wd))(flattened)

    wd = base_wd / (43.0 / annotations.NUM_MOD_MEDIUM)
    y_medium = keras.layers.Dense(annotations.NUM_MOD_MEDIUM,
                                  kernel_initializer="normal", activation="softmax",
                                  kernel_regularizer=keras.regularizers.l2(wd),
                                  use_bias=False,
                                  name="y_medium")(dense1)


    wd = base_wd / (43.0 / annotations.NUM_MOD_FINE)
    y_fine = keras.layers.Dense(annotations.NUM_MOD_FINE,
                                kernel_initializer="normal", activation="softmax",
                                kernel_regularizer=keras.regularizers.l2(wd),
                                use_bias=False,
                                name="y_fine")(dense1)

    return [tfr_input], [y_coarse, y_medium, y_fine]

