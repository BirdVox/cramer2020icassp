import csv
import datetime
import jams
import librosa
import h5py
import muda
import numpy as np
import os
import sys
import time

import localmodule


# Define constants.
data_dir = localmodule.get_train_data_dir()
dataset_name = localmodule.get_train_dataset_name()
args = sys.argv[1:]
aug_str = args[0]
instance_str = str(int(args[1]))


# Print header.
start_time = int(time.time())
print(str(datetime.datetime.now()) + " Start.")
print("Augmenting " + dataset_name)
print("with augmentation " + aug_str + " and instance " + instance_str + ".")
print("jams version: {:s}".format(jams.__version__))
print("librosa version: {:s}".format(librosa.__version__))
print("muda version: {:s}".format(muda.__version__))
print("numpy version: {:s}".format(np.__version__))
print("")


# Create directory for augmented clips.
original_dataset_h5_dir = os.path.join(data_dir, "hdf5", "original")
valid_data_dir = localmodule.get_valid_data_dir()
valid_dataset_name = localmodule.get_valid_dataset_name()
instanced_aug_str = "-".join([aug_str, instance_str])
aug_dataset_h5_dir = os.path.join(data_dir, "hdf5", instanced_aug_str)
os.makedirs(aug_dataset_h5_dir, exist_ok=True)


# Create directory corresponding to the recording unit.
in_h5_dir = original_dataset_h5_dir
out_h5_dir = aug_dataset_h5_dir
os.makedirs(out_h5_dir, exist_ok=True)


# Define deformers.
if aug_str == "noise":
    # Background noise deformers.
    # For each recording unit, we create a deformer which adds a negative
    # example (i.e. containing no flight call) to the current clip, weighted
    # by a randomized amplitude factor ranging between 0.1 and 0.5.
    # This does not change the label because
    # negative + negative = negative
    # and
    # positive + negative = positive.


    noise_dir = localmodule.get_noise_data_dir()
    noise_paths = []

    noise_csv_path = os.path.join(noise_dir, 'BirdVox-DCASE-20k_csv-public.csv')
    with open(noise_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['hasbird'] == '1':
                continue
            audio_path = os.path.join(noise_dir, "wav", row['itemid'] + '.wav')
            noise_paths.append(audio_path)

    deformer = muda.deformers.BackgroundNoise(
        n_samples=1, files=noise_paths,
        weight_min=0.0625, weight_max=0.25)

elif aug_str == "pitch":
    # Pitch shift deformer.
    # For every clip to be augmented, we apply a pitch shift whose interval
    # is sampled from a normal distribution with null mean and unit variance,
    # as measured in semitones according to the 12-tone equal temperament.
    deformer = muda.deformers.RandomPitchShift(
        n_samples=1, mean=0.0, sigma=1.0)
elif aug_str == "stretch":
    # Time stretching deformer.
    # For every clip to be augmented, we apply a time stretching whose factor
    # are sampled from a log-normal distribution with mu=0.0 and sigma=0.1.
    deformer = muda.deformers.RandomTimeStretch(
        n_samples=1, location=0.0, scale=0.1)
else:
    raise ValueError('Invalid augmentation: {}'.format(aug_str))


def create_jams(clip_name, in_path, audio, sr):
    jam = jams.JAMS()

    # Create annotation.
    ann = jams.Annotation('tag_open', sandbox={'level': 'taxonomy_code'})
    ann.duration = len(audio) / float(sr)

    origin_name, taxonomy_code, _= os.path.splitext(os.path.basename(in_path))[0].split('_')

    # Add tag with snippet sound class.
    ann.append(time=0, duration=0, value=taxonomy_code, confidence=1)

    # Fill file metadata.
    jam.file_metadata.title = clip_name
    jam.file_metadata.release = '1.0'
    jam.file_metadata.duration = ann.duration
    jam.file_metadata.artist = origin_name

    # Fill annotation metadata.
    ann.annotation_metadata.version = '1.0'
    ann.annotation_metadata.corpus = dataset_name

    # Add annotation.
    jam.annotations.append(ann)

    return jam


for fname in os.listdir(original_dataset_h5_dir):
    in_path = os.path.join(original_dataset_h5_dir, fname)
    out_fname = os.path.basename(in_path).replace('original',
        aug_str + '-' + instance_str)
    out_path = os.path.join(aug_dataset_h5_dir, out_fname)
    if os.path.exists(out_path):
        continue
    with h5py.File(in_path, 'r') as f_in:
        with h5py.File(out_path, 'w') as f_out:
            f_out["sample_rate"] = localmodule.get_sample_rate()
            waveform_group = f_out.create_group("waveforms")
            for clip_name, data in f_in['waveforms'].items():
                jam_in = create_jams(clip_name, in_path, data.value.flatten(), f_in['sample_rate'].value)
                jam_in = muda.jam_pack(jam_in,
                                       _audio=dict(y=data.value.flatten(),
                                       sr=localmodule.get_sample_rate()))

                # Apply data augmentation.
                jam_tf = deformer.transform(jam_in)

                # Get jam from jam iterator. The iterator has only one element.
                jam_out = next(jam_tf)

                # Add audio to new h5 file
                waveform_group[clip_name] = jam_out.sandbox.muda._audio.pop('y')
                #waveform_group[clip_name] = jam_out.sandbox.muda.pop('_audio')
                # Add augmentation parameters to attrs
                for k, v in localmodule.flatten_dict(jam_out.sandbox.muda).items():
                    # Skip list of all filenames in noise augmentation
                    if '_files_' in k:
                        continue
                    # Skip versions for now
                    if '_version_' in k:
                        continue
                    # Skip n_samples since it is the same for everything
                    if 'n_samples' in k:
                        continue
                    # Skip sample rate since it is always the same
                    if 'audio_sr' in k:
                        continue
                    waveform_group[clip_name].attrs['muda_' + k] = v

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
