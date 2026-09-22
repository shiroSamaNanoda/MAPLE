# MAPLE

PyTorch implementation of **MAPLE: Modality-Adaptive Prior-Guided Learning with Language-Conditioned Spatial Adaptation for Multimodal Remote Sensing Segmentation**.

MAPLE targets multimodal remote-sensing semantic segmentation with RGB/VIS imagery and DSM. The repository contains only the MAPLE implementation and the code required to train/evaluate it on the supported datasets.

## Method components

- **LMA** — language-conditioned modality-specific adapters for lightweight adaptation of the frozen SAM3 visual encoder.
- **LCAS** — language-conditioned adapter synthesis that converts vision-language similarity maps into spatially adaptive modulation parameters.
- **VLP** — RemoteCLIP-based vision-language priors used for semantic conditioning and dense classification.
- **SAFR** — spatial adaptive frequency reweighting for patch-level auxiliary supervision under class imbalance.

The SAM3 backbone and RemoteCLIP encoder are kept frozen; MAPLE trains only the lightweight adaptation, fusion, and decoding components.

## Repository structure

```text
MAPLE/
├── configs/
│   ├── maple_vaihingen.yaml
│   ├── maple_potsdam.yaml
│   └── maple_hunan.yaml
├── datasets/
│   ├── __init__.py
│   └── isprs.py
├── losses/
│   ├── __init__.py
│   └── cross_entropy.py
├── model/
│   ├── __init__.py
│   ├── base.py
│   └── maple.py
├── scripts/
│   └── prepare_hunan.py
├── src/
│   ├── config.py
│   ├── metrics.py
│   ├── optimizers.py
│   ├── registries.py
│   ├── safr_logger.py
│   ├── schedulers.py
│   ├── trainer.py
│   └── utils.py
├── train.py
├── test.py
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Installation

```bash
conda create -n maple python=3.10 -y
conda activate maple
pip install -r requirements.txt
```

MAPLE imports SAM3 with:

```python
from sam3.model.vitdet import ViT
```

Therefore, install the SAM3 codebase (or place it on `PYTHONPATH`) before training.

## Pretrained models

The default YAML files expect:

```text
pretrained/
├── sam3.pt
└── RemoteCLIP-ViT-B-32.pt
```

`pretrained/` is intentionally excluded from Git by `.gitignore`; do not commit large pretrained checkpoints to the repository.

## Datasets

Three presets are included in `datasets/isprs.py`:

- **ISPRS Vaihingen**
- **ISPRS Potsdam**
- **MMHunan**

Set `dataset.params.data_root` in the corresponding YAML file to your local dataset path.

For MMHunan preprocessing:

```bash
python scripts/prepare_hunan.py \
  --input /path/to/Hunan_Dataset \
  --output /path/to/Hunan
```

> Note: the Hunan preset references `data/hunan_train_ids.txt` and `data/hunan_test_ids.txt`. Add the split files used by your experiments before releasing a fully reproducible Hunan setup.

## Training

Vaihingen:

```bash
python train.py --config configs/maple_vaihingen.yaml
```

Potsdam:

```bash
python train.py --config configs/maple_potsdam.yaml
```

MMHunan:

```bash
python train.py --config configs/maple_hunan.yaml
```

Configuration values can be overridden from the command line, for example:

```bash
python train.py --config configs/maple_vaihingen.yaml \
  --set training.batch_size=2 optimizer.base_lr=0.005
```

## Evaluation

```bash
python test.py \
  --config configs/maple_vaihingen.yaml \
  --weights ./weights/Vaihingen/epoch50_0.8637_best.pth \
  --output-dir ./predictions
```

The `weights/`, `predictions/`, `vis_safr/`, `logs/`, and local dataset/checkpoint directories are excluded from Git.

## Configuration

Each experiment YAML specifies the model, dataset, loss, optimizer, scheduler, and training settings. MAPLE is registered under:

```yaml
model:
  name: maple
```

The default segmentation loss is registered as:

```yaml
loss:
  name: cross_entropy
```

## License

This code is released under the MIT License. See [LICENSE](LICENSE).

## Acknowledgements

This implementation builds on SAM3, RemoteCLIP, OpenCLIP, and related open-source remote-sensing segmentation work. Please follow the licenses and citation requirements of the corresponding upstream projects.
