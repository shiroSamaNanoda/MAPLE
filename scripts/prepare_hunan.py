"""Convert the raw MMHunan dataset to the PNG layout expected by MAPLE.

The raw release ships Sentinel-2 TIFFs (12 bands), topographic TIFFs and
IGBP land-cover TIFFs. This script:
    1. extracts an RGB composite from S2 bands 4/3/2,
    2. writes the topographic data as a single-channel DSM PNG,
    3. remaps the IGBP labels to the 7-class scheme used by MAPLE.

Usage:
    python scripts/prepare_hunan.py \\
        --input  /path/to/Hunan_Dataset \\
        --output /path/to/Hunan
"""
import argparse
import os
from collections import Counter
from typing import List

import numpy as np
import tifffile as tiff
from PIL import Image


# IGBP value -> MAPLE class id
LABEL_REMAP = {
    0: -1,   # ignore
    1:  0,   # cropland
    2:  1,   # forest
    3:  2,   # grassland
    5:  3,   # wetland
    6:  4,   # water
    8:  5,   # unused land
    9:  6,   # built-up area
}

LABEL_NAMES = {
    -1: 'Ignore',  0: 'Cropland', 1: 'Forest',  2: 'Grassland',
     3: 'Wetland', 4: 'Water',    5: 'Unused',  6: 'Built-up',
}

LABEL_COLORS = {
    -1: (0,   0,   0),
     0: (196, 90,  17),    1: (51,  129, 88),    2: (177, 205, 61),
     3: (228, 84,  96),    4: (91,  154, 214),   5: (225, 174, 110),
     6: (239, 159, 2),
}


# ============================================================
#  Per-modality processors
# ============================================================

def _normalise_band(band: np.ndarray) -> np.ndarray:
    """Min-max normalise to uint8 [0, 255]."""
    band = band.astype(np.float32)
    lo, hi = band.min(), band.max()
    if hi - lo < 1e-8:
        return np.zeros_like(band, dtype=np.uint8)
    return ((band - lo) / (hi - lo) * 255.0).astype(np.uint8)


def s2_to_rgb(s2_path: str) -> np.ndarray:
    """Sentinel-2 (12-band) -> uint8 RGB. Uses bands 4/3/2 (R/G/B)."""
    s2 = tiff.imread(s2_path)
    if s2.ndim != 3 or s2.shape[-1] < 4:
        raise ValueError(f"Unexpected S2 shape: {s2.shape}")
    r = _normalise_band(s2[..., 3])
    g = _normalise_band(s2[..., 2])
    b = _normalise_band(s2[..., 1])
    return np.stack([r, g, b], axis=-1)


def topo_to_dsm(topo_path: str) -> np.ndarray:
    """Topo TIFF -> uint8 single-channel DSM."""
    arr = tiff.imread(topo_path)
    if arr.ndim == 3:
        arr = arr[0] if arr.shape[0] == 2 else arr[..., 0]
    return _normalise_band(arr)


def remap_label(label: np.ndarray) -> np.ndarray:
    """IGBP -> MAPLE class id; unknowns become -1."""
    out = np.full_like(label, fill_value=-1, dtype=np.int8)
    for src, dst in LABEL_REMAP.items():
        out[label == src] = dst
    return out


def label_to_rgb(label: np.ndarray) -> np.ndarray:
    h, w = label.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for value, color in LABEL_COLORS.items():
        rgb[label == value] = color
    return rgb


# ============================================================
#  Conversion driver
# ============================================================

class HunanDatasetConverter:
    def __init__(self, input_root: str, output_root: str,
                 save_visualisations: bool = False):
        self.input_root = input_root
        self.output_root = output_root
        self.save_vis = save_visualisations
        self._mkdirs()

    def _mkdirs(self):
        for sub in ('images_png', 'dsm_pngs', 'masks_png'):
            os.makedirs(os.path.join(self.output_root, sub), exist_ok=True)
        if self.save_vis:
            for sub in ('rgb_vis', 'dsm_vis', 'mask_vis'):
                os.makedirs(os.path.join(self.output_root, 'visualization', sub),
                            exist_ok=True)

    @staticmethod
    def _strip_prefix(name: str) -> str:
        for prefix in ('s2_', 'topo_', 'lc_'):
            if name.startswith(prefix):
                return name[len(prefix):]
        return name

    def _build_paths(self, split: str, s2_filename: str):
        basename = self._strip_prefix(os.path.splitext(s2_filename)[0])
        s2_path   = os.path.join(self.input_root, split, 's2',   f's2_{basename}.tif')
        topo_path = os.path.join(self.input_root, split, 'topo', f'topo_{basename}.tif')

        # Some splits use lc_<id>.tif, others <id>.tif.
        lc_with    = os.path.join(self.input_root, split, 'lc', f'lc_{basename}.tif')
        lc_without = os.path.join(self.input_root, split, 'lc', f'{basename}.tif')
        lc_path = lc_without if os.path.exists(lc_without) else lc_with

        return s2_path, topo_path, lc_path, basename

    def _process_one(self, split: str, s2_filename: str) -> bool:
        s2_path, topo_path, lc_path, basename = self._build_paths(split, s2_filename)
        for tag, path in [('s2', s2_path), ('topo', topo_path), ('lc', lc_path)]:
            if not os.path.exists(path):
                print(f"  [skip] {basename}: missing {tag} ({path})")
                return False

        rgb = s2_to_rgb(s2_path)
        Image.fromarray(rgb).save(
            os.path.join(self.output_root, 'images_png', f'{basename}.png'))

        dsm = topo_to_dsm(topo_path)
        Image.fromarray(dsm).save(
            os.path.join(self.output_root, 'dsm_pngs', f'{basename}.png'))

        label = remap_label(tiff.imread(lc_path))
        tiff.imwrite(
            os.path.join(self.output_root, 'masks_png', f'{basename}.tif'), label)

        if self.save_vis:
            vis = os.path.join(self.output_root, 'visualization')
            Image.fromarray(rgb).save(os.path.join(vis, 'rgb_vis',  f'{basename}.png'))
            Image.fromarray(dsm).save(os.path.join(vis, 'dsm_vis',  f'{basename}.png'))
            Image.fromarray(label_to_rgb(label)).save(
                os.path.join(vis, 'mask_vis', f'{basename}.png'))
        return True

    def convert(self, splits: List[str] = ('train', 'val', 'test')) -> None:
        stats = {s: [0, 0] for s in splits}     # [success, failed]
        for split in splits:
            s2_dir = os.path.join(self.input_root, split, 's2')
            if not os.path.isdir(s2_dir):
                print(f"[{split}] directory not found: {s2_dir}")
                continue
            files = sorted(f for f in os.listdir(s2_dir) if f.endswith('.tif'))
            print(f"[{split}] {len(files)} samples")
            for i, fname in enumerate(files, 1):
                ok = self._process_one(split, fname)
                stats[split][0 if ok else 1] += 1
                if i % 100 == 0:
                    print(f"  progress: {i}/{len(files)}")

        print("\n=== summary ===")
        for split, (ok, bad) in stats.items():
            print(f"  {split}: {ok} ok / {bad} failed (total {ok + bad})")

    def analyse_label_distribution(self) -> None:
        mask_dir = os.path.join(self.output_root, 'masks_png')
        if not os.path.isdir(mask_dir):
            return
        counter = Counter()
        total = 0
        for f in os.listdir(mask_dir):
            if not f.endswith('.tif'):
                continue
            arr = tiff.imread(os.path.join(mask_dir, f)).flatten()
            counter.update(arr.tolist())
            total += arr.size

        print("\n=== label distribution ===")
        print(f"{'class':<12}{'pixels':>16}{'fraction':>12}")
        print('-' * 40)
        for value in sorted(counter):
            name = LABEL_NAMES.get(int(value), f'Unknown({value})')
            count = counter[value]
            print(f"{name:<12}{count:>16,}{count/total:>12.2%}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--input',  required=True, help='Raw Hunan_Dataset root.')
    p.add_argument('--output', required=True, help='Destination root.')
    p.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    p.add_argument('--vis', action='store_true', help='Save colour visualisations.')
    args = p.parse_args()

    conv = HunanDatasetConverter(args.input, args.output,
                                 save_visualisations=args.vis)
    conv.convert(args.splits)
    conv.analyse_label_distribution()


if __name__ == '__main__':
    main()
