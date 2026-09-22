"""Logger for SAFR class-weight dynamics during training."""
import json
import os
from typing import List, Optional

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'font.size': 12, 'axes.labelsize': 13,
    'axes.titlesize': 14, 'legend.fontsize': 10, 'figure.dpi': 150,
})


class SAFRWeightLogger:
    """Records SAFR class-weight statistics during training and plots them.

    Usage (one line in each spot of `train()`):
        logger = SAFRWeightLogger(num_classes, class_names)
        logger.log(model, iteration)            # inside the train loop
        logger.save(out_dir); logger.plot(out_dir)   # after training
    """

    def __init__(self, num_classes: int,
                 class_names: Optional[List[str]] = None,
                 log_interval: int = 10):
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class {i}' for i in range(num_classes)]
        self.log_interval = log_interval
        self.history = {
            'iterations':    [],
            'class_freq':    [],
            'isf_weights':   [],
            'final_weights': [],
        }

    def log(self, model, iteration: int) -> None:
        if iteration % self.log_interval != 0:
            return
        fn = getattr(model, 'safr_loss', None) or getattr(model, 'aux_loss_fn', None)
        if fn is None or not hasattr(fn, 'class_freq'):
            return

        freq = fn.class_freq.cpu().numpy().copy()
        weights = fn.class_weights.cpu().numpy().copy()

        eps = 1e-6
        isf = np.log((1.0 + eps) / (freq + eps))
        isf_n = (isf - isf.min()) / (isf.max() - isf.min() + eps)
        isf_n = 0.5 + isf_n

        self.history['iterations'].append(iteration)
        self.history['class_freq'].append(freq)
        self.history['isf_weights'].append(isf_n)
        self.history['final_weights'].append(weights)

    # ----- serialization ------------------------------------------------

    def save(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        d = {
            'class_names': self.class_names,
            'num_classes': self.num_classes,
            'iterations':    self.history['iterations'],
            'class_freq':    [x.tolist() for x in self.history['class_freq']],
            'isf_weights':   [x.tolist() for x in self.history['isf_weights']],
            'final_weights': [x.tolist() for x in self.history['final_weights']],
        }
        path = os.path.join(output_dir, 'safr_weight_history.json')
        with open(path, 'w') as f:
            json.dump(d, f, indent=2)
        print(f"SAFR history saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'SAFRWeightLogger':
        with open(path) as f:
            d = json.load(f)
        obj = cls(d['num_classes'], d['class_names'])
        obj.history['iterations']    = d['iterations']
        obj.history['class_freq']    = [np.array(x) for x in d['class_freq']]
        obj.history['isf_weights']   = [np.array(x) for x in d['isf_weights']]
        obj.history['final_weights'] = [np.array(x) for x in d['final_weights']]
        return obj

    # ----- plotting -----------------------------------------------------

    def plot(self, output_dir: str = './vis_safr', fmt: str = 'pdf') -> None:
        os.makedirs(output_dir, exist_ok=True)
        if not self.history['iterations']:
            print("No data recorded; nothing to plot.")
            return

        iters = np.array(self.history['iterations'])
        w   = np.array(self.history['final_weights'])
        f   = np.array(self.history['class_freq'])
        isf = np.array(self.history['isf_weights'])

        self._curve(iters, w,   'SAFR Class Weights During Training',     'Class Weight',
                    os.path.join(output_dir, f'safr_weights_curve.{fmt}'))
        self._curve(iters, f,   'EMA-Smoothed Class Frequency',           'Spatial Frequency',
                    os.path.join(output_dir, f'safr_frequency_curve.{fmt}'), invert=True)
        self._curve(iters, isf, 'Inverse Spatial Frequency (ISF) Weights', 'ISF Weight',
                    os.path.join(output_dir, f'safr_isf_curve.{fmt}'))
        self._combined(iters, w, f,
                       os.path.join(output_dir, f'safr_combined.{fmt}'))
        self._bar(w[-1], os.path.join(output_dir, f'safr_final_bar.{fmt}'))
        print(f"All SAFR figures saved under {output_dir}")

    # ----- internals ----------------------------------------------------

    def _styles(self):
        c = plt.cm.Set1(np.linspace(0, 1, self.num_classes))
        m = ['o', 's', '^', 'D', 'v', 'P', '*']
        ls = ['-', '--', '-.', ':', '-', '--', '-.']
        return c, m, ls

    def _curve(self, iters, data, title, ylabel, path, invert=False):
        co, mk, ls = self._styles()
        fig, ax = plt.subplots(figsize=(10, 5))
        mi = max(1, len(iters) // 15)
        for c in range(self.num_classes):
            ax.plot(iters, data[:, c], color=co[c], linestyle=ls[c % 7],
                    linewidth=2, label=self.class_names[c],
                    marker=mk[c % 7], markevery=mi, markersize=6, alpha=.85)
        ax.set_xlabel('Training Iteration')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        h, la = ax.get_legend_handles_labels()
        if invert:
            h, la = h[::-1], la[::-1]
        ax.legend(h, la, loc='best', framealpha=.9, edgecolor='gray')
        ax.grid(True, alpha=.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight', pad_inches=.1)
        plt.close()

    def _combined(self, iters, weights, freqs, path):
        co, mk, _ = self._styles()
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 5.5))
        mi = max(1, len(iters) // 12)

        for c in range(self.num_classes):
            a1.plot(iters, weights[:, c], color=co[c], lw=2,
                    label=self.class_names[c], marker=mk[c % 7],
                    markevery=mi, markersize=5, alpha=.85)
        a1.set_xlabel('Training Iteration')
        a1.set_ylabel('Adaptive Class Weight')
        a1.set_title('(a) SAFR Adaptive Weights', fontweight='bold')
        a1.grid(True, alpha=.3, linestyle='--')
        a1.spines['top'].set_visible(False)
        a1.spines['right'].set_visible(False)
        if len(iters) > 0:
            we = min(100, iters[-1] * 0.1)
            a1.axvline(x=we, color='gray', linestyle=':', alpha=.7)
            a1.text(we + 5, a1.get_ylim()[1] * .95, 'Warmup\nEnd',
                    fontsize=9, color='gray', va='top')

        for c in range(self.num_classes):
            a2.plot(iters, freqs[:, c], color=co[c], lw=2,
                    label=self.class_names[c], marker=mk[c % 7],
                    markevery=mi, markersize=5, alpha=.85)
        a2.set_xlabel('Training Iteration')
        a2.set_ylabel('EMA Spatial Frequency')
        a2.set_title('(b) EMA-Smoothed Class Frequency', fontweight='bold')
        a2.grid(True, alpha=.3, linestyle='--')
        a2.spines['top'].set_visible(False)
        a2.spines['right'].set_visible(False)

        h, la = a1.get_legend_handles_labels()
        fig.legend(h, la, loc='lower center', ncol=self.num_classes,
                   fontsize=11, markerscale=1.5,
                   bbox_to_anchor=(.5, -.05), frameon=False)
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight', pad_inches=.2)
        plt.close()

    def _bar(self, weights, path):
        co = plt.cm.Set1(np.linspace(0, 1, self.num_classes))
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(range(self.num_classes), weights,
                      color=co, edgecolor='white', lw=1.5, width=.6)
        for b, w in zip(bars, weights):
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + .02, f'{w:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_xticks(range(self.num_classes))
        ax.set_xticklabels(self.class_names, fontsize=11)
        ax.set_ylabel('Final SAFR Weight')
        ax.set_title('Final Adaptive Class Weights', fontweight='bold')
        ax.axhline(y=1.0, color='gray', ls='--', alpha=.5, label='Uniform')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight', pad_inches=.1)
        plt.close()
