import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="D:/C2/runs/train/exp1/results.csv")
parser.add_argument("--output", default=None)
args = parser.parse_args()
csv_path = args.csv
epochs, box_loss, cls_loss, map50, map5095, precision, recall = [], [], [], [], [], [], []

with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        row = {key.strip(): value for key, value in row.items()}
        epochs.append(int(row['epoch']))
        box_loss.append(float(row['train/box_loss']))
        cls_loss.append(float(row['train/cls_loss']))
        map50.append(float(row['metrics/mAP50(B)']))
        map5095.append(float(row['metrics/mAP50-95(B)']))
        precision.append(float(row['metrics/precision(B)']))
        recall.append(float(row['metrics/recall(B)']))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss
axes[0,0].plot(epochs, box_loss, 'b-', label='box_loss', linewidth=2)
axes[0,0].plot(epochs, cls_loss, 'r-', label='cls_loss', linewidth=2)
axes[0,0].set_xlabel('Epoch', fontsize=12)
axes[0,0].set_ylabel('Loss', fontsize=12)
axes[0,0].set_title('Training Loss', fontsize=14)
axes[0,0].legend(fontsize=11)
axes[0,0].grid(True, alpha=0.3)

# mAP
axes[0,1].plot(epochs, map50, 'b-', label='mAP50', linewidth=2)
axes[0,1].plot(epochs, map5095, 'r-', label='mAP50-95', linewidth=2)
best_idx = max(range(len(map5095)), key=map5095.__getitem__)
best_epoch = epochs[best_idx]
axes[0,1].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best (epoch {best_epoch})')
axes[0,1].set_xlabel('Epoch', fontsize=12)
axes[0,1].set_ylabel('mAP', fontsize=12)
axes[0,1].set_title('mAP Metrics', fontsize=14)
axes[0,1].legend(fontsize=11)
axes[0,1].grid(True, alpha=0.3)

# P/R
axes[1,0].plot(epochs, precision, 'g-', label='Precision', linewidth=2)
axes[1,0].plot(epochs, recall, 'orange', label='Recall', linewidth=2)
axes[1,0].set_xlabel('Epoch', fontsize=12)
axes[1,0].set_ylabel('Value', fontsize=12)
axes[1,0].set_title('Precision & Recall', fontsize=14)
axes[1,0].legend(fontsize=11)
axes[1,0].grid(True, alpha=0.3)

# Summary text
axes[1,1].axis('off')
summary = f"""
Final Results (Best Epoch: {best_epoch})
{'='*40}
Overall:
  mAP50:      {max(map50):.3f}
  mAP50-95:   {max(map5095):.3f}
  Precision:  {precision[best_idx]:.3f}
  Recall:     {recall[best_idx]:.3f}

Training:
  Total epochs: {len(epochs)}
  Final box_loss: {box_loss[-1]:.3f}
  Final cls_loss: {cls_loss[-1]:.3f}
"""
axes[1,1].text(0.1, 0.5, summary, fontsize=12, fontfamily='monospace',
               verticalalignment='center', transform=axes[1,1].transAxes,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
out_path = args.output or str(Path(csv_path).with_name("results_plot.png"))
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved to {out_path}")
