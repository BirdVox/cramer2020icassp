import csv
import os
import re
import oyaml as yaml
import datetime
from operator import itemgetter
from collections import Counter

import localmodule


ANNOTATION_FIELDS = [
    'Origin',
    'Recording Date',
    'Selection',
    'Clip Name',
    'View',
    'Channel',
    'Clip Date',
    'Clip Time',
    'Begin Time (s)',
    'End Time (s)',
    'Low Freq (Hz)',
    'High Freq (Hz)',
    'Begin Path',
    'Taxonomy Code',
    'Fine Label',
    'Medium Label',
    'Coarse Label',
]


def normalize_text(string):
    return re.sub("[^\\w]", "", string.lower())


def get_taxonomy_mapping(taxonomy):
    mapping = {}
    for coarse_dict in taxonomy.values():
        for medium_dict in coarse_dict.values():
            for taxonomy_code, labels in medium_dict.items():
                for label in labels:
                    label = normalize_text(label)
                    mapping[label] = taxonomy_code

    return mapping


def get_taxonomy_idxs(taxonomy):
    coarse_codes = []
    medium_codes = []
    fine_codes = []

    for coarse_code, medium_dict in taxonomy.items():
        coarse_code = str(coarse_code)
        if 'X' not in coarse_code:
            coarse_codes.append(coarse_code)
        for medium_code, fine_dict in medium_dict.items():
            medium_code = str(medium_code)
            if 'X' not in medium_code:
                medium_codes.append(medium_code)
            for fine_code in fine_dict.keys():
                fine_code = str(fine_code)
                if 'X' not in fine_code:
                    fine_codes.append(fine_code)

    coarse_idxs = {code: idx for idx, code in enumerate(sorted(coarse_codes))}
    medium_idxs = {code: idx for idx, code in enumerate(sorted(medium_codes))}
    fine_idxs = {code: idx for idx, code in enumerate(sorted(fine_codes))}

    return coarse_idxs, medium_idxs, fine_idxs


def get_taxonomy_code_idx_triplet(taxonomy_code):
    coarse_idx = COARSE_IDXS.get(taxonomy_code.split('.')[0])
    medium_idx = MEDIUM_IDXS.get(taxonomy_code.rsplit('.', 1)[0])
    fine_idx = FINE_IDXS.get(taxonomy_code)

    return coarse_idx, medium_idx, fine_idx


def get_modified_taxonomy_idxs(taxonomy, filter_dict):
    coarse_idx = 0
    medium_idx = 0
    fine_idx = 0

    coarse_idxs = {}
    medium_idxs = {}
    fine_idxs = {}

    for coarse_code, medium_dict in taxonomy.items():
        coarse_code = str(coarse_code)
        if coarse_code in filter_dict['coarse'].get('ignore', []):
            continue
        if 'X' not in coarse_code:
            if coarse_code not in filter_dict['coarse'].get('other', []):
                coarse_idxs[coarse_code] = coarse_idx
                coarse_idx += 1
            else:
                coarse_idxs[coarse_code] = -1

        for medium_code, fine_dict in medium_dict.items():
            medium_code = str(medium_code)
            if medium_code in filter_dict['medium'].get('ignore', []):
                continue
            if 'X' not in medium_code:
                if medium_code not in filter_dict['medium'].get('other', []):
                    medium_idxs[medium_code] = medium_idx
                    medium_idx += 1
                else:
                    medium_idxs[medium_code] = -1

            for fine_code in fine_dict.keys():
                fine_code = str(fine_code)
                if fine_code in filter_dict['fine'].get('ignore', []):
                    continue
                if 'X' not in fine_code:
                    if fine_code not in filter_dict['fine'].get('other', []):
                        fine_idxs[fine_code] = fine_idx
                        fine_idx += 1
                    else:
                        fine_idxs[fine_code] = -1

    # Set all other classes to map to the last index
    for k in coarse_idxs.keys():
        if coarse_idxs[k] == -1:
            coarse_idxs[k] = coarse_idx
    for k in medium_idxs.keys():
        if medium_idxs[k] == -1:
            medium_idxs[k] = medium_idx
    for k in fine_idxs.keys():
        if fine_idxs[k] == -1:
            fine_idxs[k] = fine_idx

    return coarse_idxs, medium_idxs, fine_idxs


def get_modified_taxonomy_medium_children_count(taxonomy, filter_dict):
    medium_counts = {}

    for coarse_code, medium_dict in taxonomy.items():
        coarse_code = str(coarse_code)
        if coarse_code in filter_dict['coarse'].get('ignore', []) or coarse_code in filter_dict['coarse'].get('other', []) or 'X' in coarse_code:
            continue

        for medium_code, fine_dict in medium_dict.items():
            medium_code = str(medium_code)
            if medium_code in filter_dict['medium'].get('ignore', []) or medium_code in filter_dict['medium'].get('other', []) or 'X' in medium_code:
                continue
            medium_counts[medium_code] = 0

            for fine_code in fine_dict.keys():
                fine_code = str(fine_code)
                if fine_code in filter_dict['fine'].get('ignore', []) or fine_code in filter_dict['fine'].get('other', []) or 'X' in medium_code:
                    continue
                medium_counts[medium_code] += 1

    return medium_counts


def get_modified_taxonomy_code_idx_triplet(taxonomy_code):
    coarse_idx = MOD_COARSE_IDXS.get(taxonomy_code.split('.')[0])
    medium_idx = MOD_MEDIUM_IDXS.get(taxonomy_code.rsplit('.', 1)[0])
    fine_idx = MOD_FINE_IDXS.get(taxonomy_code)

    return coarse_idx, medium_idx, fine_idx


def load_taxonomy_codes(filepath):
    with open(filepath, 'r') as f:
        taxonomy_codes = yaml.load(f)

    return taxonomy_codes


TAXONOMY_PATH = os.path.join(os.path.dirname(__file__),
                             '..',
                             'resources',
                             'taxonomy.yaml')
FILTER_PATH = os.path.join(os.path.dirname(__file__), 'filter.yaml')

with open(TAXONOMY_PATH, 'r') as f:
    TAXONOMY = yaml.load(f)

with open(FILTER_PATH, 'r') as f:
    FILTER = yaml.load(f)

MAPPINGS = get_taxonomy_mapping(TAXONOMY)

COARSE_IDXS, MEDIUM_IDXS, FINE_IDXS = get_taxonomy_idxs(TAXONOMY)
MOD_COARSE_IDXS, MOD_MEDIUM_IDXS, MOD_FINE_IDXS = get_modified_taxonomy_idxs(TAXONOMY, FILTER)
MOD_MEDIUM_COUNTS = get_modified_taxonomy_medium_children_count(TAXONOMY, FILTER)

NUM_COARSE = len(COARSE_IDXS)
NUM_MEDIUM = len(MEDIUM_IDXS)
NUM_FINE = len(FINE_IDXS)

NUM_MOD_COARSE = len(set(MOD_COARSE_IDXS.values()))
NUM_MOD_MEDIUM = len(set(MOD_MEDIUM_IDXS.values()))
NUM_MOD_FINE = len(set(MOD_FINE_IDXS.values()))



def get_level_code(code, level):
    return '.'.join(code.split('.')[:level])


def get_level_subcode(code, level):
    return code.split('.')[level-1]


def resolve_codes(fine_code, medium_code, coarse_code):
    if medium_code == 'X.X.X':
        medium_code = coarse_code

    if fine_code == "X.X.X" \
      or (get_level_subcode(fine_code, 2) == "X"
          and get_level_subcode(fine_code, 1) == get_level_subcode(medium_code, 1)):
        fine_code = medium_code

    if coarse_code == 'X.X.X':
        coarse_code = medium_code

    fine_l2 = get_level_code(fine_code, 2)
    medium_l2 = get_level_code(medium_code, 2)
    if medium_l2.split('.')[-1] != 'X' and fine_l2 != medium_l2:
        err_msg = "Mismatch between fine and medium annotations: fine - {}, medium - {}, coarse - {}"
        raise ValueError(err_msg.format(fine_code, medium_code, coarse_code))

    fine_l1 = get_level_code(fine_code, 1)
    coarse_l1 = get_level_code(coarse_code, 1)
    if coarse_l1.split('.')[-1] != 'X' and fine_l1 != coarse_l1:
        err_msg = "Mismatch between fine and coarse annotations: fine - {}, medium - {}, coarse - {}"
        raise ValueError(err_msg.format(fine_code, medium_code, coarse_code))

    return fine_code


def get_code(fine, medium, coarse):
    fine = normalize_text(fine)
    medium = normalize_text(medium)
    coarse = normalize_text(coarse)

    if coarse in ("callnotfc", "nonfc", "nonflightcallunknown"):
        return "X.X.X"

    try:
        fine_code = MAPPINGS[fine]
        medium_code = MAPPINGS[medium]
        coarse_code = MAPPINGS[coarse]
    except KeyError as e:
        err_msg = " *** {} (Original strings: fine - {}, medium - {}, coarse - {}"
        print(err_msg.format(str(e), fine, medium, coarse))
        code = "X.X.X"
        return code


    try:
        code = resolve_codes(fine_code, medium_code, coarse_code)
    except ValueError as e:
        err_msg = " *** {} (Original strings: fine - {}, medium - {}, coarse - {}"
        #DEBUG
        print(err_msg.format(str(e), fine, medium, coarse))
        code = "X.X.X"

    return code


class BirdVoxAnnotation(object):
    def __init__(self, item, origin_name=None, recording_date=None):
        try:
            self.index = int(item['Selection'])
            self.view = item['View']
            self.channel = int(item['Channel'])
            self.start_time = float(item['Begin Time (s)'])
            self.end_time = float(item['End Time (s)'])
            self.low_freq = float(item['Low Freq (Hz)'])
            self.high_freq = float(item['High Freq (Hz)'])
            self.audio_path = item.get('Begin Path')
            self.fine = item.get('Species', '') or item.get('Fine Label', '')
            self.medium = item.get('Family', '') or item.get('Medium Label', '')
            self.coarse = item.get('Order', '') or item.get('Coarse Label', '')
            self.taxonomy_code = item.get('Taxonomy Code') \
                                 or get_code(self.fine, self.medium, self.coarse)
        except KeyError as e:
            raise ValueError('Missing field: {}; fields: {}'.format(str(e), list(item.keys())))

        self.origin_name = origin_name or item.get('Origin', '')
        self.recording_date = recording_date or item['Recording Date']
        if not self.recording_date:
            raise ValueError('Did not provide recording date.')

    @property
    def ave_freq(self):
        return int((self.low_freq + self.high_freq)/2.0)

    @property
    def timestamp(self):
        return int((self.start_time + self.end_time)/2.0 * localmodule.get_sample_rate())

    @property
    def datetime(self):
        offset = datetime.timedelta(seconds=(self.start_time + self.end_time)/2.0)
        if len(self.recording_date) <= 8:
            print("Missing time from recording date.")
            return datetime.datetime.strptime(self.recording_date, "%Y%m%d") + offset

        return datetime.datetime.strptime(self.recording_date, "%Y%m%d%H%M%S") + offset

    @property
    def time(self):
        return self.datetime.strftime("%H-%M-%S")

    @property
    def date(self):
        return self.datetime.strftime("%Y-%m-%d")

    @property
    def clip_name(self):
        # <origin_name>-<recording_date>_<center_timestamp>_<ave_freq>
        return "{}-{}_{:09}_{:05}".format(self.origin_name,
                                          self.recording_date,
                                          self.timestamp,
                                          self.ave_freq)

    @property
    def duration_seconds(self):
        return self.end_time - self.start_time

    @property
    def duration_samples(self):
        return int(self.duration_seconds * localmodule.get_sample_rate())

    def get_output_dict(self, audio_dir=None):
        clip_name = self.clip_name
        if audio_dir:
            begin_path = os.path.join(audio_dir, self.origin_name, clip_name + ".wav")
        elif self.audio_path:
            begin_path = self.audio_path
        else:
            begin_path = self.origin_name, clip_name + ".wav"

        return {
            'Origin': self.origin_name,
            'Recording Date': self.recording_date,
            'Selection': self.index,
            'Clip Name': clip_name,
            'View': self.view,
            'Channel': self.channel,
            'Clip Date': self.date,
            'Clip Time': self.time,
            'Begin Time (s)': self.start_time,
            'End Time (s)': self.end_time,
            'Low Freq (Hz)': self.low_freq,
            'High Freq (Hz)': self.low_freq,
            'Begin Path': begin_path,
            'Taxonomy Code': self.taxonomy_code,
            'Fine Label': self.fine,
            'Medium Label': self.medium,
            'Coarse Label': self.coarse,
        }


class BirdVoxUnit(object):
    def __init__(self, origin_name, annotations):
        self.origin_name = origin_name
        self.files = {}

        for ann in annotations:
            clip_name = ann.clip_name
            self.files[clip_name] = ann

    def get_annotation(self, clip_name):
        return self.files[clip_name]

    def save_annotations(self, annotations_dir, audio_dir=None):
        rows = sorted([v.get_output_dict(audio_dir)
                       for k,v in self.files.items()], key=itemgetter('Selection'))

        annotations_path = os.path.join(annotations_dir, self.origin_name + '.txt')
        with open(annotations_path, 'w') as f:
            writer = csv.DictWriter(f, ANNOTATION_FIELDS, delimiter='\t')
            writer.writeheader()
            writer.writerows(rows)

    def __len__(self):
        return len(self.files)

    def get_histograms(self):
        fine_hist = Counter()
        medium_hist = Counter()
        coarse_hist = Counter()
        for ann in self.files.values():
            fine_hist[ann.fine] += 1
            medium_hist[ann.medium] += 1
            coarse_hist[ann.coarse] += 1

        return fine_hist, medium_hist, coarse_hist


class BirdVoxAnnotationDataset(object):
    def __init__(self, annotation_dir):
        self.units = {}

        unit_annotations = {}

        # Load all annotations in directory
        for fname in os.listdir(annotation_dir):
            if not (fname.endswith('.txt') or fname.endswith('.csv')):
                continue

            print("Loading annotations from {}...".format(fname))
            path = os.path.join(annotation_dir, fname)

            origin_name = self.get_origin_name(fname)
            if origin_name not in unit_annotations:
                unit_annotations[origin_name] = []

            # Aggregate recordings for units across different dates
            unit_annotations[origin_name] += self.load_annotation_file(path)

        for origin_name, ann_list in unit_annotations.items():
            self.units[origin_name] = BirdVoxUnit(origin_name, ann_list)

    @property
    def fine_labels(self):
        """Return fine in canonical coarse"""
        fine = set()

        for unit in self.units.values():
            for ann in unit.files.values():
                fine.add(ann.fine)

        return list(sorted(fine))

    @property
    def medium_labels(self):
        """Return families in canonical coarse"""
        mediums = set()

        for unit in self.units.values():
            for ann in unit.files.values():
                mediums.add(ann.medium)

        return list(sorted(mediums))

    @property
    def coarse_labels(self):
        """Return families in canonical coarse"""
        coarses = set()

        for unit in self.units.values():
            for ann in unit.files.values():
                coarses.add(ann.coarse)

        return list(sorted(coarses))

    def __len__(self):
        return sum([len(unit) for unit in self.units.values()])

    def get_annotation(self, clip_name):
        clip_name = os.path.splitext(clip_name)[0]
        origin_name = self.get_origin_name(clip_name)

        return self.units[origin_name].get_annotation(clip_name)

    @staticmethod
    def get_origin_name(fname):
        unit_start_idx = fname.find("unit") + 4

        # Account for when the unit is specified as "unit-01" instead of "unit01"
        while not fname[unit_start_idx].isdigit():
            unit_start_idx += 1

        return "unit{}".format(fname[unit_start_idx:unit_start_idx+2])

    @staticmethod
    def load_annotation_file(filepath):
        # Get recording date (and time) if it is available
        fname = os.path.basename(filepath)
        stripped_path = fname.replace('-', '')
        origin_name = BirdVoxAnnotationDataset.get_origin_name(fname)
        path_parts = stripped_path.split('_')
        date_parts = [x for x in path_parts if x.startswith('2015')]
        bv70k_pattern = r'unit\d{2,2}_\d{9,9}_\d{5,5}_[01]\.Table\.1\.selections\.txt'
        if len(date_parts) > 0:
            recording_date = date_parts[0]
        elif re.match(bv70k_pattern, stripped_path):
            # If BV70k:
            offset = int(path_parts[1]) / float(localmodule.get_sample_rate())
            offset_delta = datetime.timedelta(seconds=offset)
            unit_str = path_parts[0]
            start_time_path = os.path.join(os.path.dirname(filepath), '../..',
                'BirdVox-70k_utc-start-times.csv')

            with open(start_time_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['Unit'] == unit_str:
                        ts = int(row['UTC'].strip())
                        start_dt = datetime.datetime.utcfromtimestamp(ts)

            dt = start_dt + offset_delta
            recording_date = dt.strftime("%Y%m%d%H%M%S")

        else:
            recording_date = ""


        annotations = []
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                annotations.append(BirdVoxAnnotation(row, origin_name=origin_name,
                                                     recording_date=recording_date))

        return annotations

    def save_annotations(self, annotations_dir, audio_dir=None):
        for unit in self.units.values():
            unit.save_annotations(annotations_dir, audio_dir=audio_dir)

    def get_histograms(self):
        fine_hist = Counter()
        medium_hist = Counter()
        coarse_hist = Counter()
        for unit in self.units.values():
            unit_fine_hist, unit_medium_hist, unit_coarse_hist \
                = unit.get_histograms()

            fine_hist += unit_fine_hist
            medium_hist += unit_medium_hist
            coarse_hist += unit_coarse_hist

        return fine_hist, medium_hist, coarse_hist

    def get_annotation_iterator(self):
        for unit in self.units.values():
            for ann in unit.files.values():
                yield ann

    def get_taxonomy_code_to_ann_map(self):
        dct = {}
        for ann in self.get_annotation_iterator():
            if ann.taxonomy_code not in dct:
                dct[ann.taxonomy_code] = []

            dct[ann.taxonomy_code].append(ann)

        return dct
