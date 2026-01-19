#!/usr/bin/env python3

import json
import sys
from pathlib import Path



import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

RUNS_DIR = Path("runs/task1/transfer_learning")
OUTPUT_DIR = Path("docs/figs/task1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

runs = []
for run_dir in sorted(RUNS_DIR.iterdir()):
    if not run_dir.is_dir():
        continue
    
    metrics_file = run_dir / "metrics.json"
    if not metrics_file.exists():
        continue
    
    with open(metrics_file) as f:
        data = json.load(f)
    
    runs.append({
        'name': run_dir.name,
        'data': data,
        'pretrained': data.get('pretrained', False),
        'mode': data['config']['freeze_mode'],
        'epochs': [e['epoch'] for e in data['epochs']],
        'val_acc': [e['val_acc'] * 100 for e in data['epochs']],
    })

runs.sort(key=lambda x: (not x['pretrained'], x['mode']))

label_map = {
    (True, 'head_only'): 'Pretrained (Head Only)',
    (True, 'final_block'): 'Pretrained (Final Block)',
    (False, 'head_only'): 'Random (Head Only)',
    (False, 'final_block'): 'Random (Final Block)',
}

color_map = {
    (True, 'head_only'): '#2E86AB',
    (True, 'final_block'): '#A23B72',
    (False, 'head_only'): '#F18F01',
    (False, 'final_block'): '#C73E1D',
}

linestyle_map = {
    'head_only': '-',
    'final_block': '--',
}

fig, ax = plt.subplots(figsize=(8, 5))

for run in runs:
    key = (run['pretrained'], run['mode'])
    label = label_map[key]
    color = color_map[key]
    linestyle = linestyle_map[run['mode']]
    
    ax.plot(run['epochs'], run['val_acc'], 
            label=label, 
            color=color, 
            linestyle=linestyle,
            linewidth=2.5,
            alpha=0.9)

ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
ax.legend(loc='lower right', framealpha=0.95, fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_xlim(1, 30)
ax.set_ylim(0, 85)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

output_file = OUTPUT_DIR / "transfer_learning_comparison.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Saved to: {output_file}")

print("\nSummary:")
for run in runs:
    key = (run['pretrained'], run['mode'])
    best_acc = max(run['val_acc'])
    best_epoch = run['epochs'][run['val_acc'].index(best_acc)]
    print(f"{label_map[key]:30s}: {best_acc:5.2f}% (epoch {best_epoch})")
