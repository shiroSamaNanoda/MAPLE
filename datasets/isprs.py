"""ISPRS / MMHunan dataset.

Registered as ``'isprs'``.  Supports three named presets via
``params.preset``: ``Vaihingen``, ``Potsdam``, ``Hunan``.  Each preset
fixes the file-name patterns, class labels, palette, train/test ID
split, and eval stride.

YAML:

    dataset:
      name: isprs
      params:
        preset: Vaihingen
        data_root: /data/ISPRS
        split: train                     # 'train' or 'test'
        window_size: [256, 256]
        cache: true
        augmentation: true
        length_multiplier: 1000          # virtual epoch size

The same class instance answers both:

- ``__getitem__`` -> a training crop (dict batch), used by the DataLoader.
- ``eval_iter()`` -> a generator that yields one full test tile at a
  time for sliding-window inference (used by ``src/trainer.evaluate``).

The dataset also exposes everything the trainer needs to know about the
task: ``num_classes``, ``labels``, ``text_prompts``, ``palette``,
``stride_size``, ``window_size``, ``drop_last_class_in_eval``.  The
trainer never reads a "config" object -- it asks the dataset directly.
"""
from __future__ import annotations

import os
import random
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from skimage import io

from src.registries import DATASET_REGISTRY
from src.utils import convert_from_color, get_random_pos


# ============================================================
#  Built-in presets
# ============================================================
#
# Each preset captures: per-tile file-name patterns, IDs to use for
# train and test, the class taxonomy, palette, and standard eval
# stride.  Add a new preset here, or pass full params in the YAML and
# omit ``preset`` entirely.

PRESETS: Dict[str, dict] = {
    'Vaihingen': {
        'train_ids': [
            '1', '3', '23', '26', '7', '11',
            '13', '28', '17', '32', '34', '37',
        ],
        'test_ids': ['5', '21', '15', '30'],
        'stride_size': 32,
        'data_pattern':    'top/top_mosaic_09cm_area{}.tif',
        'dsm_pattern':     'dsm/dsm_09cm_matching_area{}.tif',
        'label_pattern':   'gts_for_participants/top_mosaic_09cm_area{}.tif',
        'eroded_pattern':  'gts_eroded_for_participants/'
                           'top_mosaic_09cm_area{}_noBoundary.tif',
        'labels': ["roads", "buildings", "low veg.", "trees", "cars", "clutter"],
        'text_prompts': ["impervious surface", "building", "low vegetation",
                         "tree", "car", "clutter background"],
        'palette': {
            0: (255, 255, 255), 1: (0, 0, 255),   2: (0, 255, 255),
            3: (0, 255, 0),     4: (255, 255, 0), 5: (255, 0, 0),
            6: (0, 0, 0),
        },
        'subfolder': 'Vaihingen/',
        'drop_last_class_in_eval': True,
        'has_nir_to_drop': False,
        'eroded_is_rgb':   True,
    },

    'Potsdam': {
        'train_ids': [
            '6_10', '7_10', '2_12', '3_11', '2_10', '7_8',
            '5_10', '3_12', '5_12', '7_11', '7_9',  '6_9',
            '7_7',  '4_12', '6_8',  '6_12', '6_7',  '4_11',
        ],
        'test_ids': ['4_10', '5_11', '2_11', '3_10', '6_11', '7_12'],
        'stride_size': 128,
        'data_pattern':   '4_Ortho_RGBIR/top_potsdam_{}_RGBIR.tif',
        'dsm_pattern':    '1_DSM_normalisation/'
                          'dsm_potsdam_{}_normalized_lastools.jpg',
        'label_pattern':  '5_Labels_for_participants/'
                          'top_potsdam_{}_label.tif',
        'eroded_pattern': '5_Labels_for_participants_no_Boundary/'
                          'top_potsdam_{}_label_noBoundary.tif',
        'labels': ["roads", "buildings", "low veg.", "trees", "cars", "clutter"],
        'text_prompts': ["impervious surface", "building", "low vegetation",
                         "tree", "car", "clutter background"],
        'palette': {
            0: (255, 255, 255), 1: (0, 0, 255),   2: (0, 255, 255),
            3: (0, 255, 0),     4: (255, 255, 0), 5: (255, 0, 0),
            6: (0, 0, 0),
        },
        'subfolder': 'Potsdam/',
        'drop_last_class_in_eval': True,
        'has_nir_to_drop': True,        # RGBIR -> drop NIR channel
        'eroded_is_rgb':   True,
    },

    'Hunan': {
        'train_ids_file': 'data/hunan_train_ids.txt',
        'test_ids_file':  'data/hunan_test_ids.txt',
        'stride_size': 256,
        'data_pattern':   'images_png/{}.png',
        'dsm_pattern':    'dsm_pngs/{}.png',
        'label_pattern':  'masks_png/{}.tif',
        'eroded_pattern': 'masks_png/{}.tif',
        'labels': ["cropland", "forest", "grassland", "wetland",
                   "water", "unused land", "built-up area"],
        'text_prompts': ["background", "building", "road", "water",
                         "barren", "forest", "agricultural land"],
        'palette': {
            0: (196, 90, 17),   1: (51, 129, 88),  2: (177, 205, 61),
            3: (228, 84, 96),   4: (91, 154, 214), 5: (225, 174, 110),
            6: (239, 159, 2),
        },
        'subfolder': '',
        'drop_last_class_in_eval': False,
        'has_nir_to_drop': False,
        'eroded_is_rgb':   False,        # masks already indexed
    },
}


def _load_id_list(path: str) -> List[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _augment(*arrays, flip: bool = True, mirror: bool = True):
    """Identical random flips across all arrays."""
    do_flip   = flip   and random.random() < 0.5
    do_mirror = mirror and random.random() < 0.5
    out = []
    for a in arrays:
        if do_flip:
            a = a[..., ::-1, :] if a.ndim >= 2 else a
        if do_mirror:
            a = a[..., :, ::-1] if a.ndim >= 2 else a
        out.append(np.ascontiguousarray(a))
    return tuple(out)


# ============================================================
#  Dataset class
# ============================================================

@DATASET_REGISTRY.register('isprs')
class ISPRSDataset(torch.utils.data.Dataset):
    """ISPRS Vaihingen / Potsdam / MMHunan, picked by ``preset``."""

    def __init__(self,
                 preset: str,
                 data_root: str,
                 split: str = 'train',
                 window_size: Tuple[int, int] = (256, 256),
                 cache: bool = True,
                 augmentation: bool = True,
                 length_multiplier: Optional[int] = None,
                 # Advanced: override individual preset fields without
                 # creating a new preset entry.
                 overrides: Optional[Dict[str, Any]] = None):
        super().__init__()
        if preset not in PRESETS:
            raise ValueError(
                f"Unknown ISPRS preset {preset!r}; "
                f"choose from {list(PRESETS)}")

        raw = dict(PRESETS[preset])
        if overrides:
            raw.update(overrides)

        # ---- resolve paths ----
        root = data_root.rstrip('/') + '/' + raw['subfolder']
        self.data_pattern    = root + raw['data_pattern']
        self.dsm_pattern     = root + raw['dsm_pattern']
        self.label_pattern   = root + raw['label_pattern']
        self.eroded_pattern  = root + raw['eroded_pattern']

        # ---- resolve IDs ----
        if 'train_ids' in raw:
            self.train_ids = list(raw['train_ids'])
            self.test_ids  = list(raw['test_ids'])
        else:
            self.train_ids = _load_id_list(raw['train_ids_file'])
            self.test_ids  = _load_id_list(raw['test_ids_file'])

        # ---- public task metadata (read by the trainer) ----
        self.preset                  = preset
        self.labels: List[str]       = list(raw['labels'])
        self.text_prompts: List[str] = list(raw['text_prompts'])
        self.palette                 = raw['palette']
        self.invert_palette          = {v: k for k, v in self.palette.items()}
        self.window_size             = tuple(window_size)
        self.stride_size             = raw['stride_size']
        self.drop_last_class_in_eval = raw['drop_last_class_in_eval']
        self._has_nir_to_drop        = raw['has_nir_to_drop']
        self._eroded_is_rgb          = raw['eroded_is_rgb']
        self.name                    = preset

        # ---- training-time state ----
        self.split = split
        if split not in ('train', 'test'):
            raise ValueError(f"split must be 'train' or 'test' (got {split!r})")
        ids = self.train_ids if split == 'train' else self.test_ids

        self.augmentation = augmentation
        self.cache = cache
        self._data_cache:  dict = {}
        self._dsm_cache:   dict = {}
        self._label_cache: dict = {}

        self.data_files  = [self.data_pattern.format(i)   for i in ids]
        self.dsm_files   = [self.dsm_pattern.format(i)    for i in ids]
        self.label_files = [self.label_pattern.format(i)  for i in ids]
        for f in self.data_files + self.dsm_files + self.label_files:
            if not os.path.isfile(f):
                raise FileNotFoundError(f)

        if length_multiplier is None:
            length_multiplier = 500 if preset == 'Hunan' else 1000
        self._length = length_multiplier

    # ----- standard dataset metadata ---------------------------------

    @property
    def num_classes(self) -> int:
        return len(self.labels)

    # ----- low-level IO with caching ---------------------------------

    def _load_rgb(self, idx: int) -> np.ndarray:
        if idx in self._data_cache:
            return self._data_cache[idx]
        arr = io.imread(self.data_files[idx])
        if self._has_nir_to_drop:
            arr = arr[:, :, :3]
        arr = (arr.transpose(2, 0, 1).astype(np.float32) / 255.0)
        if self.cache:
            self._data_cache[idx] = arr
        return arr

    def _load_dsm(self, idx: int) -> np.ndarray:
        if idx in self._dsm_cache:
            return self._dsm_cache[idx]
        dsm = io.imread(self.dsm_files[idx]).astype(np.float32)
        lo, hi = dsm.min(), dsm.max()
        dsm = (dsm - lo) / max(hi - lo, 1e-8)
        if self.cache:
            self._dsm_cache[idx] = dsm
        return dsm

    def _load_label(self, idx: int) -> np.ndarray:
        if idx in self._label_cache:
            return self._label_cache[idx]
        raw = io.imread(self.label_files[idx])
        if self._eroded_is_rgb:
            label = convert_from_color(raw, self.invert_palette).astype(np.int64)
        else:
            label = raw.astype(np.int64)
        if self.cache:
            self._label_cache[idx] = label
        return label

    # ----- main entry: training crop ---------------------------------

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, _: int) -> Dict[str, torch.Tensor]:
        i = random.randint(0, len(self.data_files) - 1)
        data  = self._load_rgb(i)
        dsm   = self._load_dsm(i)
        label = self._load_label(i)

        if self.preset == 'Hunan':
            data_p, dsm_p, label_p = data, dsm, label
        else:
            x1, x2, y1, y2 = get_random_pos(data, self.window_size)
            data_p  = data[:, x1:x2, y1:y2]
            dsm_p   = dsm[x1:x2, y1:y2]
            label_p = label[x1:x2, y1:y2]

        if self.augmentation:
            data_p, dsm_p, label_p = _augment(data_p, dsm_p, label_p)

        return {
            'image': torch.from_numpy(data_p),
            'dsm':   torch.from_numpy(dsm_p),
            'label': torch.from_numpy(label_p),
        }

    # ----- evaluation: yield full tiles ------------------------------

    def eval_iter(self) -> Iterator[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
        """Yield ``(image_id, full_rgb, full_dsm, gt_eroded)`` for each
        test tile.  Used by the framework's sliding-window evaluator
        (``src/trainer.evaluate``)."""
        for img_id in self.test_ids:
            img = io.imread(self.data_pattern.format(img_id))
            if self._has_nir_to_drop:
                img = img[:, :, :3]
            img = img.astype(np.float32) / 255.0

            dsm = io.imread(self.dsm_pattern.format(img_id)).astype(np.float32)
            lo, hi = dsm.min(), dsm.max()
            dsm = (dsm - lo) / max(hi - lo, 1e-8)

            raw = io.imread(self.eroded_pattern.format(img_id))
            if self._eroded_is_rgb:
                gt_eroded = convert_from_color(raw, self.invert_palette)
            else:
                gt_eroded = raw.astype(np.int64)

            yield img_id, img, dsm, gt_eroded
