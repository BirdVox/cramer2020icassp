import datetime
import h5py
import librosa
import os
import sys
import time

import localmodule


# Define constants.
dataset_name = localmodule.get_train_dataset_name()
sample_rate = localmodule.get_sample_rate()
args = sys.argv[1:]
data_dir = args[0]
aug_str = args[1]
instance_id = int(args[2])
instance_str = str(instance_id)
if aug_str == "original":
    instanced_aug_str = aug_str
else:
    instanced_aug_str = "-".join([aug_str, instance_str])
pcen_settings = localmodule.get_pcen_settings()


# Print header.
start_time = int(time.time())
print(str(datetime.datetime.now()) + " Start.")
print("Computing per-channel energy normalization (PCEN for clips in " + data_dir)
print("Augmentation: " + instanced_aug_str + ".")
print("")
print("h5py version: {:s}".format(h5py.__version__))
print("librosa version: {:s}".format(librosa.__version__))
print("")


# Open HDF5 container of waveforms.
hdf5_dir = os.path.join(data_dir, "hdf5")
in_aug_dir = os.path.join(hdf5_dir, instanced_aug_str)

# Create HDF5 container of PCENs.
pcen_dir = os.path.join(data_dir, "pcen")
os.makedirs(pcen_dir, exist_ok=True)
out_aug_dir = os.path.join(pcen_dir, instanced_aug_str)
os.makedirs(out_aug_dir, exist_ok=True)

for fname in os.listdir(in_aug_dir):
    in_path = os.path.join(in_aug_dir, fname)
    print("* Computing PCEN for {}...".format(fname))
    out_path = os.path.join(out_aug_dir, fname)

    #if os.path.exists(out_path):
    #    continue

    with h5py.File(in_path, "r") as in_file:
        try:
            sample_rate = in_file["sample_rate"].value
        except KeyError:
            sample_rate = localmodule.get_sample_rate()

        with h5py.File(out_path, "w") as out_file:
            # Copy over metadata.
            out_file["dataset_name"] = localmodule.get_train_dataset_name()
            out_file["augmentation"] = aug_str
            out_file["instance"] = instance_id
            settings_group = out_file.create_group("pcen_settings")
            settings_group["fmax"] = pcen_settings["fmax"]
            settings_group["fmin"] = pcen_settings["fmin"]
            settings_group["hop_length"] = pcen_settings["hop_length"]
            settings_group["n_fft"] = pcen_settings["n_fft"]
            settings_group["n_mels"] = pcen_settings["n_mels"]
            settings_group["sr"] = pcen_settings["sr"]
            settings_group["win_length"] = pcen_settings["win_length"]
            settings_group["window"] = pcen_settings["window"]
            settings_group["pcen_delta"] = pcen_settings["pcen_delta"]
            settings_group["pcen_time_constant"] = pcen_settings["pcen_time_constant"]
            settings_group["pcen_norm_exponent"] = pcen_settings["pcen_norm_exponent"]
            settings_group["pcen_power"] = pcen_settings["pcen_power"]

            # List clips.
            lms_group = out_file.create_group("pcen")
            clip_names = list(in_file["waveforms"].keys())

            # Loop over clips.
            for clip_name in clip_names:
                # Load waveform.
                waveform = in_file["waveforms"][clip_name].value.flatten()

                waveform = (waveform * (2**31)).astype('float32')

                # Resample to 22050 Hz.
                if sample_rate != pcen_settings["sr"]:
                    waveform = librosa.resample(
                        waveform, sample_rate, pcen_settings["sr"])



                # Compute Short-Term Fourier Transform (STFT).
                stft = librosa.stft(
                    waveform,
                    n_fft=pcen_settings["n_fft"],
                    win_length=pcen_settings["win_length"],
                    hop_length=pcen_settings["hop_length"],
                    window=pcen_settings["window"])

                # Compute squared magnitude coefficients.
                abs2_stft = (stft.real*stft.real) + (stft.imag*stft.imag)

                # Gather frequency bins according to the Mel scale.
                melspec = librosa.feature.melspectrogram(
                    y=None,
                    S=abs2_stft,
                    sr=pcen_settings["sr"],
                    n_fft=pcen_settings["n_fft"],
                    n_mels=pcen_settings["n_mels"],
                    htk=True,
                    fmin=pcen_settings["fmin"],
                    fmax=pcen_settings["fmax"])

                # Compute PCEN.
                pcen = librosa.pcen(melspec,
                    sr=pcen_settings["sr"],
                    hop_length=pcen_settings["hop_length"],
                    gain=pcen_settings["pcen_norm_exponent"],
                    bias=pcen_settings["pcen_delta"],
                    power=pcen_settings["pcen_power"],
                    time_constant=pcen_settings["pcen_time_constant"])

                # Convert to single floating-point precision.
                pcen = pcen.astype('float32')

                # Truncate spectrum to range 2-10 kHz.
                pcen = pcen[:pcen_settings["top_freq_id"], :]

                # Save.
                lms_group[clip_name] = pcen

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
