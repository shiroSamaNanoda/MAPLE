# MAPLE

PyTorch implementation of our TGRS 2026 paper **MAPLE: Modality-Adaptive Prior-Guided Learning with Language-Conditioned Spatial Adaptation for Multimodal Remote Sensing Segmentation**.



## Installation

```bash
conda create -n maple python=3.10 -y
conda activate maple
pip install -r requirements.txt
```


## Pretrained Models

Place the pretrained checkpoints in:

```text
pretrained/
├── sam3.pt
└── RemoteCLIP-ViT-B-32.pt
```

## Datasets

Supported datasets:

- ISPRS Vaihingen
- ISPRS Potsdam
- MMHunan

Set the dataset path in the corresponding YAML file under `configs/`.

## Training

```bash
# Vaihingen
python train.py --config configs/maple_vaihingen.yaml

# Potsdam
python train.py --config configs/maple_potsdam.yaml

# MMHunan
python train.py --config configs/maple_hunan.yaml
```

## Evaluation

```bash
python test.py \
  --config configs/maple_vaihingen.yaml \
  --weights ./weights/Vaihingen/best.pth \
  --output-dir ./predictions
```

## License

This project is released under the MIT License.

## Acknowledgements

This implementation builds on SAM3, RemoteCLIP, OpenCLIP, and related open-source remote-sensing segmentation projects.
