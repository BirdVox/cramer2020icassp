import argparse
import csv
import datetime
import pickle as pk
import h5py
import keras
import numpy as np
import os
import pandas as pd
import pescador
import tensorflow as tf
import time
import oyaml as yaml
import localmodule
from models import create_flat_singletask_medium_model
from keras.optimizers import Adam


# Define constants.
train_data_dir = localmodule.get_train_data_dir()
train_dataset_name = localmodule.get_train_dataset_name()
valid_data_dir = localmodule.get_valid_data_dir()
valid_dataset_name = localmodule.get_valid_dataset_name()
models_dir = localmodule.get_models_dir()
n_input_hops = 104
n_filters = [24, 48, 48]
kernel_size = [5, 5]
pool_size = [2, 4]
n_hidden_units = 64

# Read command-line arguments.
parser = argparse.ArgumentParser()
parser.add_argument('aug_kind_str')
parser.add_argument('trial_str')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--base-wd', type=float, default=1e-4)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--epochs', type=int)
parser.add_argument('--tfr', default="pcen")
parser.add_argument('--align-perturb', action='store_true')
parser.add_argument('--lr-annealing', action='store_true')
parser.add_argument('--num-cpus', type=int, default=1)
args = parser.parse_args()

aug_kind_str = args.aug_kind_str
trial_str = args.trial_str
lr = args.lr
base_wd = args.base_wd
batch_size = args.batch_size
epochs = args.epochs
tfr_str = args.tfr
num_cpus = args.num_cpus
align_perturb = args.align_perturb
lr_annealing = args.lr_annealing


active_streamers = 64
streamer_rate = 1024
steps_per_epoch = int(np.ceil(36.0 * localmodule.get_num_augmentations(aug_kind_str) * streamer_rate / batch_size))

# Iterate over the entire training validation set
valid_batch_size = 512
validation_steps = int(np.ceil(35335 / float(valid_batch_size)))


# Set number of epochs.
if not epochs:
    if aug_kind_str == "none":
        epochs = 512
    else:
        epochs = 1024


# Print header.
start_time = int(time.time())
print(str(datetime.datetime.now()) + " Start.")
print("Training flat single-task (medium) convnet on " + train_dataset_name)
print("")
print('h5py version: {:s}'.format(h5py.__version__))
print('keras version: {:s}'.format(keras.__version__))
print('numpy version: {:s}'.format(np.__version__))
print('pandas version: {:s}'.format(pd.__version__))
print('pescador version: {:s}'.format(pescador.__version__))
print('tensorflow version: {:s}'.format(tf.__version__))
print("")


# Define and compile Keras model.
# NB: the original implementation of Justin Salamon in ICASSP 2017 relies on
# glorot_uniform initialization for all layers, and the optimizer is a
# stochastic gradient descent (SGD) with a fixed learning rate of 0.1.
# Instead, we use a he_normal initialization for the layers followed
# by rectified linear units (see He ICCV 2015), and replace the SGD by
# the Adam adaptive stochastic optimizer (see Kingma ICLR 2014).
# Moreover, we disable dropout because we found that it consistently prevented
# the model to train at all.


inputs, outputs \
    = create_flat_singletask_medium_model(n_input_hops, n_filters, n_hidden_units, kernel_size, pool_size, base_wd=base_wd)

# Build Pescador streamers corresponding to log-mel-spectrograms in augmented
# training and validation sets.
valid_hdf5_dir = os.path.join(valid_data_dir, tfr_str, "original")


# Create directory for model, unit, and trial.
model_name = "classify-flat-singletask-medium-convnet"
if not aug_kind_str == "none":
    model_name = "_".join([model_name, "aug-" + aug_kind_str])
model_dir = os.path.join(models_dir, model_name)
os.makedirs(model_dir, exist_ok=True)
trial_dir = os.path.join(model_dir, trial_str)
os.makedirs(trial_dir, exist_ok=True)


# Define Keras callback for checkpointing model.
network_name = "_".join(
    [train_dataset_name, model_name, trial_str, "network"])
network_path = os.path.join(trial_dir, network_name + ".hdf5")
checkpoint = keras.callbacks.ModelCheckpoint(network_path,
    monitor="val_loss", verbose=False, save_best_only=True, mode="min")


# Save configuration
params_yaml_path = os.path.join(trial_dir, network_name + "-params.yaml")
with open(params_yaml_path, "w") as yaml_file:
    params = vars(args)
    params.update({
        "n_input_hops": n_input_hops,
        "n_filters": n_filters,
        "kernel_size": kernel_size,
        "pool_size": pool_size,
        "n_hidden_units": n_hidden_units,
        "active_streamers": active_streamers,
        "streamer_rate": streamer_rate,
        "steps_per_epoch": steps_per_epoch,
        "valid_batch_size": valid_batch_size,
        "validation_steps": validation_steps,
    })
    yaml.dump(params, yaml_file)


# Define custom callback for saving history.
history_name = "_".join(
    [train_dataset_name, model_name, trial_str, "history"])
history_path = os.path.join(trial_dir, history_name + ".csv")
with open(history_path, 'w') as csv_file:
    csv_writer = csv.writer(csv_file)
    header = [
        "Epoch", "Local time",
        "Training loss", "Training medium accuracy (%)",
        "Validation loss", "Validation medium accuracy (%)"]
    csv_writer.writerow(header)
def write_row(history_path, epoch, logs):
    with open(history_path, 'a') as csv_file:
        csv_writer = csv.writer(csv_file)
        row = [
            str(epoch).zfill(3),
            str(datetime.datetime.now()),
            "{:.16f}".format(logs.get('loss')),
            "{:.3f}".format(100*logs.get('acc')).rjust(7),
            "{:.16f}".format(logs.get('val_loss')),
            "{:.3f}".format(100*logs.get('val_acc')).rjust(7)]
        csv_writer.writerow(row)
history_callback = keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: write_row(history_path, epoch, logs))

callbacks = [checkpoint, history_callback]

if lr_annealing:
    reduce_lr = keras.callbacks.ReduceLROnPlateau()
    callbacks.append(reduce_lr)

print("Loading validation data.")
# Load validation data once to save on IO costs
validation_data = localmodule.get_validation_data(
        valid_hdf5_dir, n_input_hops, valid_batch_size, validation_steps,
        tfr_str=tfr_str, structured=False, label_inputs=False, single_output="medium")

print("Performing rejection sampling for initialization.")
# Rejection sampling for best initialization.
n_inits = 10
for init_id in range(n_inits):
    model = keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
        optimizer=Adam(lr), metrics=["accuracy"])

    training_streamer = localmodule.multiplex_tfr(
        train_data_dir, n_input_hops, batch_size, mode="train",
        aug_kind_str=aug_kind_str, tfr_str=tfr_str,
        partial_labels=False, structured=False, single_output="medium",
        active_streamers=active_streamers, streamer_rate=streamer_rate,
        num_cpus=num_cpus, align_perturb=align_perturb)

    history = model.fit_generator(
        training_streamer,
        steps_per_epoch = steps_per_epoch,
        epochs = 4,
        verbose = False,
        callbacks = [history_callback],
        workers=0,
        validation_data=validation_data,
        use_multiprocessing=True,
        max_queue_size=100)
    history_df = pd.read_csv(history_path)
    val_acc = 100 * list(history_df["Validation medium accuracy (%)"])[-1]
    if val_acc > 20.0:
        break


# Export network architecture as YAML file.
yaml_path = os.path.join(trial_dir, network_name + ".yaml")
with open(yaml_path, "w") as yaml_file:
    yaml_string = model.to_yaml()
    yaml_file.write(yaml_string)


# Print model summary.
model.summary()

print("Starting training.")
# Train model.
history = model.fit_generator(
    training_streamer,
    steps_per_epoch = steps_per_epoch,
    epochs = epochs,
    verbose = False,
    callbacks = callbacks,
    workers=0,
    validation_data=validation_data,
    use_multiprocessing=True,
    max_queue_size=100)


# Save some results for error analysis
results_path = os.path.join(trial_dir, network_name + "-results.pkl")
results = {
    'train': localmodule.get_results_output(model, training_streamer, 64, structured=False),
    'valid': localmodule.get_results_output(model, validation_data, 64, structured=False)
}
with open(results_path, 'wb') as f:
    pk.dump(results, f)


# Print history.
history_df = pd.DataFrame(history.history)
print(history_df.to_string())
print("")


# Print elapsed time.
print(str(datetime.datetime.now()) + " Finish.")
elapsed_time = time.time() - int(start_time)
elapsed_hours = int(elapsed_time / (60 * 60))
elapsed_minutes = int((elapsed_time % (60 * 60)) / 60)
elapsed_seconds = elapsed_time % 60.
elapsed_str = "{:>02}:{:>02}:{:>05.2f}".format(elapsed_hours,
                                               elapsed_minutes,
                                               elapsed_seconds)
print("Total elapsed time: " + elapsed_str + ".")
