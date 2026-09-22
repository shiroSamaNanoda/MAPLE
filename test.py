"""Generic YAML-driven evaluation / inference entry point.

    python test.py --config configs/maple_vaihingen.yaml \\
        --weights ./weights/Vaihingen/epoch50_0.8637_best.pth \\
        --output-dir ./predictions
"""
import argparse
import os
from pathlib import Path

import torch
from skimage import io

import model       # noqa: F401  triggers MODEL_REGISTRY auto-discovery
import datasets    # noqa: F401  triggers DATASET_REGISTRY auto-discovery

from model    import build_model
from datasets import build_dataset
from src      import convert_to_color, evaluate, load_experiment_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a registered model.")
    p.add_argument('--config',     required=True, help='Training YAML.')
    p.add_argument('--weights',    required=True, help='Trained .pth.')
    p.add_argument('--output-dir', default='./predictions',
                   help='Where to write coloured prediction PNGs.')
    p.add_argument('--stride',     type=int, default=None,
                   help='Sliding-window stride; defaults to dataset preset.')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--gpu',        default='0')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    exp = load_experiment_config(args.config)

    # Test split only; no augmentation, no length multiplier needed for eval.
    ds_params = dict(exp.dataset.get('params', {}))
    eval_ds = build_dataset(exp.dataset['name'], split='test',
                            augmentation=False, **ds_params)

    model_params = dict(exp.model.get('params', {}))
    model_params.setdefault('num_classes', eval_ds.num_classes)
    net = build_model(exp.model['name'], **model_params).cuda()

    state = torch.load(args.weights, map_location='cpu', weights_only=True)
    net.load_state_dict(state, strict=False)
    net.eval()
    print(f"[load] {args.weights}")

    miou, all_preds, _, ids = evaluate(
        net, eval_ds, stride=args.stride, batch_size=args.batch_size,
        return_predictions=True,
    )
    print(f"[result] {eval_ds.name} mIoU = {miou:.4f}")

    out_dir = Path(args.output_dir) / eval_ds.name
    out_dir.mkdir(parents=True, exist_ok=True)
    for pred, img_id in zip(all_preds, ids):
        rgb = convert_to_color(pred, eval_ds.palette)
        io.imsave(out_dir / f"inference_{img_id}.png", rgb)
    print(f"[done] saved {len(all_preds)} predictions to {out_dir}")


if __name__ == '__main__':
    main()
