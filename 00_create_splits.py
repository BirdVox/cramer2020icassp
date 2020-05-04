from ortools.algorithms import pywrapknapsack_solver
import h5py
import os
from collections import Counter
from scipy.stats import entropy as kl_div
from numpy.linalg import norm
import shutil
import numpy as np
import localmodule


def get_dataset_distribution(hdf5_dir):
    distr = {}
    for fname in os.listdir(hdf5_dir):
        source, code = fname.split('_')[:2]
        if 'X' in code:
            continue

        code = code.replace('-', '.')

        if source not in distr:
            distr[source] = Counter()

        hdf5_path = os.path.join(hdf5_dir, fname)
        with h5py.File(hdf5_path, 'r') as f:
            if code.startswith("2") or '0' in code:
                if "other" not in distr[source]:
                    distr[source]["other"] = 0
                distr[source]["other"] += len(f['waveforms'])
            else:
                distr[source][code] = len(f['waveforms'])
    return distr


def JSD(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (kl_div(_P, _M) + kl_div(_Q, _M))


# Setup directories
anafcc_dir = localmodule.get_anafcc_dir()
bv14sd_dir = localmodule.get_birdvox14sd_dir()
anafcc_distr = get_dataset_distribution(anafcc_dir)
bv14sd_distr = get_dataset_distribution(bv14sd_dir)

anafcc_counts = {k: sum(v.values()) for k,v in anafcc_distr.items()}
total_weight = sum(anafcc_counts.values())
# Knapsack search space
valid_ratio_min = 0.15
valid_ratio_max = 0.3
targets = np.arange(int(total_weight * valid_ratio_min), int(total_weight * valid_ratio_max)).tolist()

source_count_list = []
source_name_list = []
for source, count in anafcc_counts.items():
    source_name_list.append(source)
    source_count_list.append(count)
weights = [source_count_list]

# Create the solver.
solver = pywrapknapsack_solver.KnapsackSolver(
    pywrapknapsack_solver.KnapsackSolver.KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER,
    'test')

# Solve knapsack for each target weight
valid_candidates = []
ratio_list = []
for target in targets:
    values = weights[0]
    capacities = [target]

    solver.Init(values, weights, capacities)
    computed_value = solver.Solve()

    packed_items = [x for x in range(0, len(weights[0]))
                    if solver.BestSolutionContains(x)]
    ratio = sum([values[x] for x in packed_items]) / float(total_weight)
    if valid_ratio_min <= ratio <= valid_ratio_max:
        valid_candidates.append(packed_items)
        ratio_list.append(ratio)

# Remove duplicates
valid_candidates = list(set([tuple(x) for x in valid_candidates]))

# Compute test set distribution
keys = sorted(bv14sd_distr['BirdVox-300h'].keys())
bv14sd_distr_arr = np.array([bv14sd_distr['BirdVox-300h'][k] for k in keys]).astype(float)
bv14sd_distr_arr /= bv14sd_distr_arr.sum()

# Find candidate split with minimum JS-divergence
min_error = float('inf')
min_valid_idxs = None
min_valid_distr = None
min_train_distr = None
for valid_idxs in valid_candidates:
    train_distr = Counter()
    valid_distr = Counter()

    for idx in valid_idxs:
        source = source_name_list[idx]
        distr = anafcc_distr[source]
        valid_distr += distr

    for idx in range(len(source_name_list)):
        if idx in valid_idxs:
            continue
        source = source_name_list[idx]
        distr = anafcc_distr[source]
        train_distr += distr

    valid_distr_arr = np.array([valid_distr[k] for k in keys]).astype(float)
    valid_distr_arr /= valid_distr_arr.sum()
    train_distr_arr = np.array([valid_distr[k] for k in keys]).astype(float)
    train_distr_arr /= train_distr_arr.sum()

    valid_jsd = JSD(bv14sd_distr_arr, valid_distr_arr)
    train_jsd = JSD(bv14sd_distr_arr, train_distr_arr)

    error = max(valid_jsd, train_jsd)

    if error < min_error:
        min_error = error
        min_valid_idxs = valid_idxs
        min_valid_distr = valid_distr
        min_train_distr = train_distr

# Get list of train and validation "sources"
train_sources = []
valid_sources = []
for idx, source in enumerate(source_name_list):
    if idx in min_valid_idxs:
        valid_sources.append(source)
    else:
        train_sources.append(source)

# Copy train and validation data
train_dir = os.path.join(localmodule.get_train_data_dir(), "original")
valid_dir = os.path.join(localmodule.get_valid_data_dir(), "original")
for root, dirs, files in os.walk(anafcc_dir):
    for fname in files:
        for source in train_sources:
            if (fname.endswith('.hdf5') or fname.endswith(
                    '.h5')) and fname.startswith(source):
                src_path = os.path.join(root, fname)
                rel_path = src_path.replace(anafcc_dir, '')
                dst_path = os.path.join(train_dir, rel_path)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy(src_path, dst_path)
        for source in valid_sources:
            if (fname.endswith('.hdf5') or fname.endswith(
                    '.h5')) and fname.startswith(source):
                src_path = os.path.join(root, fname)
                rel_path = src_path.replace(anafcc_dir, '')
                dst_path = os.path.join(valid_dir, rel_path)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy(src_path, dst_path)

# Copy test data
test_dir = os.path.join(localmodule.get_test_data_dir(), "original")
for root, dirs, files in os.walk(bv14sd_dir):
    for fname in files:
        if (fname.endswith('.hdf5') or fname.endswith('.h5')):
            src_path = os.path.join(root, fname)
            rel_path = src_path.replace(bv14sd_dir, '')
            dst_path = os.path.join(test_dir, rel_path)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(src_path, dst_path)
