import glob
import h5py
import numpy as np
import os
import pescador
import random
from collections import Counter
try:
    import keras
    import keras.backend as K
except:
    pass

import annotations


def get_augmentations():
    units = get_units()
    augmentations = {
        "noise": 5,
        "original": 1,
        "pitch": 5,
        "stretch": 5,

    }
    return augmentations


def get_anafcc_dir():
    return "/beegfs/jtc440/anafcc"


def get_birdvox14sd_dir():
    return "/beegfs/jtc440/birdvox-14sd"


def get_train_data_dir():
    return "/beegfs/jtc440/birdvox-cls-train"


def get_train_dataset_name():
    return "BirdVox-cls-train"


def get_noise_data_dir():
    return "/beegfs/jtc440/birdvox-dcase-20k"


def get_valid_data_dir():
    return "/beegfs/jtc440/birdvox-cls-valid"


def get_valid_dataset_name():
    return "BirdVox-cls-valid"


def get_test_data_dir():
    return "/beegfs/jtc440/birdvox-cls-test"


def get_pcen_settings():
    pcen_settings = {
        "fmin": 2000,
        "fmax": 11025,
        "hop_length": 32,
        "n_fft": 1024,
        "n_mels": 128,
        "pcen_delta": 10.0,
        "pcen_time_constant": 0.06,
        "pcen_norm_exponent": 0.8,
        "pcen_power": 0.25,
        "sr": 22050.0,
        "top_freq_id": 120,
        "win_length": 256,
        "window": "flattop"}
    return pcen_settings


def get_logmelspec_settings():
    logmelspec_settings = {
        "fmin": 2000,
        "fmax": 11025,
        "hop_length": 32,
        "n_fft": 1024,
        "n_mels": 128,
        "sr": 22050,
        "win_length": 256,
        "window": "hann"}
    return logmelspec_settings


def flatten_dict(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(dict(y))
    return out


def get_models_dir():
    return "/scratch/jtc440/BirdVox-70k_models"


def get_sample_rate():
    return 24000 # in Hertz


def get_units():
    return ["unit" + str(unit).zfill(2) for unit in [1, 2, 3, 5, 7, 10]]


def cycle_partial_buffer_stream(mux, batch_size):
    while True:
        buffered_streamer = pescador.maps.buffer_stream(mux, batch_size, partial=True)
        for batch in buffered_streamer:
            yield batch


def get_num_augmentations(aug_kind_str):
    if aug_kind_str == "none":
        augs = 1
    elif aug_kind_str == "pitch":
        augs = 2
    elif aug_kind_str == "stretch":
        augs = 2
    elif aug_kind_str == "all-but-noise":
        augs = 3
    elif aug_kind_str == "all":
        augs = 4
    elif aug_kind_str == "noise":
        augs = 2
    else:
        raise ValueError('Invalid augmentation kind: {}'.format(aug_kind_str))
    return augs


def is_valid_data_hdf5(path, partial_labels):
    fname = os.path.basename(path)
    if not fname.endswith('.h5') and not fname.endswith('.hdf5'):
        return False

    taxonomy_code = os.path.splitext(fname)[0].split('_')[1].replace('-', '.')
    if not partial_labels and 'X' in taxonomy_code:
        return False

    # Ignore invalid codes that we want to ignore
    triplet = annotations.get_modified_taxonomy_code_idx_triplet(taxonomy_code)
    if None in triplet and ('X' not in taxonomy_code and not partial_labels):
        return False

    return True


def create_zmq_mux(streamers, num_cpus, active_streamers, streamer_rate, weights=None):
    num_streamers = len(streamers)
    if weights is None:
        weights = np.ones((num_streamers,))

    weights = np.array(weights)

    # Normalize to sum to 1
    weights = weights / weights.sum()

    partition_size = max(int(num_streamers / float(num_cpus)), 1)

    zmq_streamers = []
    zmq_weights = []

    actual_num_cpus = 0

    for idx in range(num_cpus):
        start = partition_size * idx
        stop = min(len(streamers), start + partition_size)
        if start >= stop:
            break

        actual_num_cpus += 1

        weight = sum(weights[start:stop])
        zmq_weights.append(weight)

        sub_weights = np.array(weights[start:stop]) / weight
        zmq_streamers.append(pescador.ZMQStreamer(pescador.StochasticMux(streamers[start:stop],
                                                               n_active=int(active_streamers * weight),
                                                               rate=streamer_rate,
                                                               weights=sub_weights)))

    return pescador.StochasticMux(zmq_streamers, n_active=actual_num_cpus, rate=None, weights=zmq_weights)


def multiplex_tfr(data_dir, n_hops, batch_size, mode="inference", aug_kind_str="none", tfr_str="logmelspec",
                  label_inputs=False, partial_labels=True, structured=True, active_streamers=32, streamer_rate=1024,
                  num_cpus=1, multi_label=False, align_perturb=False, single_output="fine"):
    tfr_dir = os.path.join(data_dir, tfr_str)
    streams = []

    # Parse augmentation kind string (aug_kind_str).
    if mode == "train":
        if aug_kind_str == "none":
            augs = ["original"]
        elif aug_kind_str == "pitch":
            augs = ["original", "pitch"]
        elif aug_kind_str == "stretch":
            augs = ["original", "stretch"]
        elif aug_kind_str == "all-but-noise":
            augs = ["original", "pitch", "stretch"]
        else:
            if aug_kind_str == "all":
                augs = ["original", "pitch", "stretch", "noise"]
            elif aug_kind_str == "noise":
                augs = ["original", "noise"]
            else:
                raise ValueError('Invalid augmentation kind: {}'.format(aug_kind_str))

        # Generate a Pescador streamer for every HDF5 container, that is,
        # every unit-augmentation-instance triplet.
        aug_dict = get_augmentations()
        aug_list = []
        class_list = []
        class_count = Counter()

        for aug_str in augs:
            if aug_str == "original":
                instances = [aug_str]
            else:
                n_instances = aug_dict[aug_str]
                instances = ["-".join([aug_str, str(instance_id+1)])
                    for instance_id in range(n_instances)]
            if aug_str == "noise" and tfr_str == "logmelspec":
                bias = np.float32(-17.0)
            else:
                bias = np.float32(0.0)
            for instanced_aug_str in instances:
                aug_dir = os.path.join(tfr_dir, instanced_aug_str)
                lms_name = "_".join(["*", instanced_aug_str])

                lms_pattern = os.path.join(aug_dir, lms_name + ".h5*")
                for lms_path in glob.glob(lms_pattern):
                    if not is_valid_data_hdf5(lms_path, partial_labels):
                        continue

                    taxonomy_code = os.path.splitext(os.path.basename(lms_path))[0].split('_')[1].replace('-', '.')

                    triplet = annotations.get_taxonomy_code_idx_triplet(taxonomy_code)
                    coarse_idx, medium_idx, fine_idx = triplet

                    if structured or single_output == "fine":
                        bal_idx = fine_idx
                    elif single_output == "medium":
                        bal_idx = medium_idx
                    elif single_output == "coarse":
                        bal_idx = coarse_idx
                    else:
                        raise ValueError("Invalid single output mode:{}".format(single_output))

                    class_list.append(bal_idx)
                    class_count[bal_idx] += 1

                    aug_list.append(aug_str)

                    stream = pescador.Streamer(yield_tfr, lms_path, n_hops, bias, tfr_str, mode, label_inputs, multi_label, align_perturb)
                    streams.append(stream)

        num_streamers = len(streams)
        num_fine_classes = len(class_count)
        num_aug = len([k for k in aug_dict.keys() if k != "original"])
        class_weights = {cls: (num_streamers / float(num_fine_classes * count)) for cls, count in class_count.items()}
        aug_weights = {aug: 1.0 if aug == "original" else 1.0 / num_aug for aug, n_inst in aug_dict.items()}

        # Weight examples to balance for class, such that each file is sampled from evenly per class. Additionally,
        # Balance so sampling any augmentation type (or original) is equally likely, despite the number of instances
        # per augmentation. Within augmentation types, instances are equally likely.
        weights = [class_weights[cls] * aug_weights[aug] for cls, aug in zip(class_list, aug_list)]

        # Multiplex streamers together.
        if num_cpus > 1:
            mux = create_zmq_mux(streams, num_cpus, active_streamers, streamer_rate, weights=weights)
        else:
            mux = pescador.StochasticMux(streams, n_active=active_streamers, rate=streamer_rate, weights=weights)

        # Create buffered streamer with specified batch size.
        buffered_streamer = pescador.maps.buffer_stream(mux, batch_size)
    else:
        # If not dealing with augmentations, just go through all HDF5 files
        weights = None
        bias = np.float32(0.0)
        for fname in os.listdir(data_dir):
            lms_path = os.path.join(data_dir, fname)

            if not is_valid_data_hdf5(lms_path, partial_labels):
                continue

            stream = pescador.Streamer(yield_tfr, lms_path, n_hops, bias, tfr_str, mode, label_inputs, multi_label, align_perturb)
            streams.append(stream)

        # Multiplex streamers together, but iterate exhaustively.
        mux = pescador.ChainMux(streams, mode='exhaustive')

        # Create buffered streamer with specified batch size.
        buffered_streamer = cycle_partial_buffer_stream(mux, batch_size)

    inputs = ["tfr_input"]
    if mode in ('train', 'valid') and structured and label_inputs:
        inputs += ["coarse_label_input", "medium_label_input"]

    if structured:
        outputs = ["y_coarse", "y_medium", "y_fine"]
    else:
        outputs = ["y_" + single_output]

    return pescador.maps.keras_tuples(buffered_streamer,
                                      inputs=inputs,
                                      outputs=outputs)


def multilabel_bce(y_true, y_pred):
    # Only incur loss when there is a positive label for some class (including other)
    mask = K.cast(K.sum(y_true, axis=-1) > 0, 'float32')
    # Don't incur loss on the "other" output, since we're considering only multilabel
    return keras.losses.binary_crossentropy(y_true[...,:-1], y_pred[...,:-1]) * mask


def get_joint_targets(y_coarse, y_medium, y_fine):
    # TODO: if it's worth it, make it general to more than one coarse class

    y_medium_joint = []
    y_fine_joint = []

    y_coarse_other = 1 - y_coarse

    fine_idx = 0
    for medium_code, num_children in annotations.MOD_MEDIUM_COUNTS.items():
        medium_idx = annotations.MOD_MEDIUM_IDXS[medium_code]
        med_cond_prob = y_medium[medium_idx]

        med_joint_prob = med_cond_prob * y_coarse
        y_medium_joint.append(med_joint_prob)

        for sub_idx in range(num_children):
            _fine_idx = fine_idx + sub_idx
            fine_cond_prob = y_fine[_fine_idx]
            fine_joint_prob = fine_cond_prob * med_joint_prob
            y_fine_joint.append(fine_joint_prob)

        fine_idx += num_children

    # Handle medium other
    y_medium_joint.extend([y_medium[-1] * y_coarse, y_medium[-1] * y_coarse_other])

    # Handle fine other
    for med_joint_prob in y_medium_joint[:-2]:
        y_fine_joint.append(y_fine[-1] * med_joint_prob)
    y_fine_joint.append(y_fine[-1] * med_joint_prob[-2])
    y_fine_joint.append(y_fine[-1] * med_joint_prob[-1])

    return y_coarse, np.array(y_medium_joint), np.array(y_fine_joint)


def yield_tfr(lms_path, n_hops, bias, tfr_str, mode, label_inputs, multi_label, align_perturb,
              offset_ms=None, reopen_period=128):
    taxonomy_code = os.path.splitext(os.path.basename(lms_path))[0].split('_')[1].replace('-', '.')

    triplet = annotations.get_modified_taxonomy_code_idx_triplet(taxonomy_code)
    coarse_idx, medium_idx, fine_idx = triplet

    if not multi_label:
        if annotations.NUM_MOD_COARSE > 2:
            num_coarse_outputs = annotations.NUM_MOD_COARSE
        elif annotations.NUM_MOD_COARSE == 2:
            num_coarse_outputs = 1
        else:
            raise ValueError('Invalid number of coarse classes: {}'.format(annotations.NUM_MOD_COARSE))
        num_medium_outputs = annotations.NUM_MOD_MEDIUM
        num_fine_outputs = annotations.NUM_MOD_FINE
    else:
        num_coarse_outputs = annotations.NUM_MOD_COARSE - 1
        num_medium_outputs = annotations.NUM_MOD_MEDIUM - 1
        num_fine_outputs = annotations.NUM_MOD_FINE - 1

    y_coarse = np.zeros((num_coarse_outputs,), dtype='float32')
    y_medium = np.zeros((num_medium_outputs,), dtype='float32')
    y_fine = np.zeros((num_fine_outputs,), dtype='float32')

    # Create one hot vectors, or zero vectors for unknown classes
    if coarse_idx is not None and coarse_idx < num_coarse_outputs:
        y_coarse[coarse_idx] = 1
    if medium_idx is not None and medium_idx < num_medium_outputs:
        y_medium[medium_idx] = 1
    if fine_idx is not None and fine_idx < num_fine_outputs:
        y_fine[fine_idx] = 1

    if mode == 'train':
        coarse_label_input = y_coarse
        medium_label_input = y_medium
    elif mode == 'valid':
        # For validation, we don't use teacher forcing but still have to provide zero inputs
        coarse_label_input = np.zeros((num_coarse_outputs,), dtype='float32')
        medium_label_input = np.zeros((num_medium_outputs,), dtype='float32')

    idx = 0
    done_flag = False
    while True:
        # Open HDF5 container. To avoid accumulating too much memory, close
        # and re-open file periodically
        with h5py.File(lms_path, "r") as lms_container:
            lms_group = lms_container[tfr_str]
            keys = list(lms_group.keys())
            for _ in range(reopen_period):
                # Open HDF5 group corresponding to time-freq representation (TFR).
                if len(keys) == idx:
                    done_flag = True
                    break

                if mode == "train":
                    # Pick a key uniformly as random.
                    key = random.choice(keys)
                else:
                    key = keys[idx]
                    idx += 1

                # Load TFR.
                X = lms_group[key]

                # Trim TFR in time to required number of hops.
                X_width = X.shape[1]

                first_col = int((X_width-n_hops) / 2)

                # Make sure we don't have both alignment perturbation and offset_ms on
                assert not (align_perturb and (offset_ms is not None))
                if align_perturb or offset_ms is not None:
                    # Randomly perturb the center of the window
                    fs = get_sample_rate()
                    pcen_hop = get_pcen_settings()['hop_length']
                    if align_perturb:
                        delta = min(int(fs * 25e-3) // pcen_hop , X_width//4) # min of 25ms and a quarter of the frame
                        first_col += int((2*random.random() - 1) * delta)
                    else:
                        first_col += int(fs * offset_ms * 1e-3)

                last_col = first_col + n_hops

                if 0 <= first_col < last_col <= X_width:
                    X = X[:, first_col:last_col]
                elif first_col < last_col < 0 or X_width < first_col < last_col:
                    assert not align_perturb
                    X = np.zeros((X.shape[0], n_hops, ))
                else:
                    left_pad = max(0, -first_col)
                    right_pad = max(0, last_col - X_width)

                    first_col = max(0, first_col)
                    last_col = min(X_width, last_col)

                    # Pad if not enough frames
                    X = np.pad(X[:, first_col:last_col],
                               [(0,0), (left_pad, right_pad)],
                               mode='constant')

                # Add trailing singleton dimension for Keras interoperability.
                X = X[:, :, np.newaxis]

                # Apply bias
                X = X + bias

                if X.shape != (120, n_hops, 1):
                    import pdb
                    pdb.set_trace()

                # Yield data and labels as dictionary.
                sample = dict(tfr_input=X, y_fine=y_fine)


                sample.update(dict(y_coarse=y_coarse, y_medium=y_medium))

                # Only add teacher forcing inputs for training or validation
                sample.update(dict(coarse_label_input=coarse_label_input, medium_label_input=medium_label_input))

                yield sample

        if done_flag:
            break


def get_results_output(model, data_streamer, num_steps, structured=True):
    tfr_inputs = []
    y_fine = []
    y_medium = []
    y_coarse = []
    pred_fine = []
    pred_medium = []
    pred_coarse = []

    if type(data_streamer) == tuple:
        num_steps = 1

    for _ in range(num_steps):
        if type(data_streamer) == tuple:
            X, y = data_streamer
        else:
            X, y = next(data_streamer)
        X = X[0]
        pred = model.predict(X)
        y_fine.append(y[-1])


        if structured:
            pred_coarse.append(pred[0])
            pred_medium.append(pred[1])
            pred_fine.append(pred[2])
            y_coarse.append(y[0])
            y_medium.append(y[1])
        else:
            pred_fine.append(pred)

        assert pred_fine[-1].shape == y_fine[-1].shape

        tfr_inputs.append(X)


    results = {
        'tfr_inputs': np.vstack(tfr_inputs),
        'y_fine': np.vstack(y_fine),
        'pred_fine': np.vstack(pred_fine),
    }

    if structured:

        results['y_medium'] = np.vstack(y_medium)
        results['y_coarse'] = np.vstack(y_coarse)
        results['pred_medium'] = np.vstack(pred_medium)
        results['pred_coarse'] = np.vstack(pred_coarse)

    return results


def get_validation_data(valid_hdf5_dir, n_input_hops, valid_batch_size,
                        validation_steps, tfr_str, structured, label_inputs, single_output="fine"):
    validation_streamer = iter(multiplex_tfr(
        valid_hdf5_dir, n_input_hops, valid_batch_size, mode="valid",
        tfr_str=tfr_str, partial_labels=False, structured=structured,
        label_inputs=label_inputs, single_output=single_output))

    X_aggr = None
    Y_aggr = None

    for idx, (X, Y) in enumerate(validation_streamer):
        if X_aggr is None:
            X_aggr = X
        else:
            if type(X) == list:
                X_aggr = [np.concatenate(X_pair, axis=0)
                          for X_pair in zip(X_aggr, X)]
            else:
                X_aggr = np.concatenate((X_aggr, X), axis=0)

        if Y_aggr is None:
            Y_aggr = Y
        else:
            if type(Y) == list:
                Y_aggr = [np.concatenate(Y_pair, axis=0)
                          for Y_pair in zip(Y_aggr, Y)]
            else:
                Y_aggr = np.concatenate((Y_aggr, Y), axis=0)

        if (idx + 1) == validation_steps:
            break

    return (X_aggr, Y_aggr)


def load_data(hdf5_dir, n_hops=104, bias=0.0, tfr_str='pcen', multi_label=False,
              partial_labels=False, offset_ms=None):
    tfr_inputs_arr = []
    y_fine_arr = []
    y_medium_arr = []
    y_coarse_arr = []


    for fname in os.listdir(hdf5_dir):
        lms_path = os.path.join(hdf5_dir, fname)
        if not is_valid_data_hdf5(lms_path, partial_labels):
            continue
        data = load_hdf5(lms_path, n_hops=n_hops, bias=bias, tfr_str=tfr_str,
                         multi_label=multi_label, offset_ms=offset_ms)
        tfr_inputs_arr.append(data[0])
        y_fine_arr.append(data[1])
        y_medium_arr.append(data[2])
        y_coarse_arr.append(data[3])


    tfr_inputs_arr = np.vstack(tfr_inputs_arr)
    y_fine_arr = np.vstack(y_fine_arr)
    y_medium_arr = np.vstack(y_medium_arr)
    y_coarse_arr = np.vstack(y_coarse_arr)

    return tfr_inputs_arr, y_fine_arr, y_medium_arr, y_coarse_arr


def load_hdf5(lms_path, n_hops=104, bias=0.0, tfr_str='pcen', multi_label=False,
              offset_ms=None):
    tfr_inputs_arr = []
    y_fine_arr = []
    y_medium_arr = []
    y_coarse_arr = []

    hdf5_gen = yield_tfr(lms_path, n_hops, bias, tfr_str, mode='valid',
                         label_inputs=False, multi_label=multi_label,
                         align_perturb=False,
                         offset_ms=offset_ms)

    for sample in hdf5_gen:
        tfr_inputs_arr.append(sample["tfr_input"][np.newaxis, :, :, :])
        y_fine_arr.append(sample["y_fine"])
        y_medium_arr.append(sample["y_medium"])
        y_coarse_arr.append(sample["y_coarse"])

    tfr_inputs_arr = np.vstack(tfr_inputs_arr)
    y_fine_arr = np.vstack(y_fine_arr)
    y_medium_arr = np.vstack(y_medium_arr)
    y_coarse_arr = np.vstack(y_coarse_arr)

    return tfr_inputs_arr, y_fine_arr, y_medium_arr, y_coarse_arr


def get_class_weights(hdf5_dir, level, tfr_str='pcen', partial_labels=True):
    class_counter = Counter()
    for fname in os.listdir(hdf5_dir):
        lms_path = os.path.join(hdf5_dir, fname)
        if not is_valid_data_hdf5(lms_path, partial_labels):
            continue

        taxonomy_code = os.path.splitext(fname)[0].split('_')[1].replace('-', '.')
        triplet = annotations.get_modified_taxonomy_code_idx_triplet(taxonomy_code)

        if level == "coarse":
            if annotations.NUM_MOD_COARSE == 2:
                if triplet[0] == 0:
                    cls = 1
                else:
                    cls = 0
            else:
                cls = triplet[0]
        elif level == "medium":
            cls = triplet[1]
        elif level == "fine":
            cls = triplet[2]
        else:
            raise ValueError('Invalid level: {}'.format(level))

        if cls is None:
            continue

        with h5py.File(lms_path, 'r') as lms_container:
            lms_group = lms_container[tfr_str]
            num_items = len(lms_group)

        class_counter[cls] += num_items

    max_count = max(class_counter.values())

    weights = {k: max_count / float(v) for k, v in class_counter.items()}
    return weights

