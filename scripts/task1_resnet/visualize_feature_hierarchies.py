
import argparse
import torch
import torch.nn as nn
from torchvision.models import resnet152, ResNet152_Weights
import numpy as np
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parents[2]))

from src.data.cifar10 import get_cifar10_loaders
from src.utils.repro import set_seed
from src.utils.logger import make_run_dir
from src.models.feature_extractor import FeatureExtractor

# Configuration
LAYER_GROUPS = {
    'resnet152': {
        'early': 'layer1.2',
        'middle': 'layer2.7',
        'late': 'layer4.2'
    }
}

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

def get_layer_groups(model_type: str = 'resnet152') -> Dict[str, str]:
    if model_type not in LAYER_GROUPS:
        raise ValueError(f"Unknown model type: {model_type}")
    return LAYER_GROUPS[model_type]

def apply_dimensionality_reduction(
    features: np.ndarray, 
    method: str = 'tsne', 
    n_components: int = 2, 
    random_state: int = 42
) -> np.ndarray:
    
    if method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=random_state, 
                      perplexity=30, max_iter=1000, verbose=1)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=random_state,
                           n_neighbors=15, min_dist=0.1)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"Applying {method.upper()} to features of shape {features.shape}...")
    return reducer.fit_transform(features)

def plot_feature_space(
    reduced_features: np.ndarray, 
    labels: np.ndarray, 
    title: str, 
    save_path: Path, 
    class_names: Optional[List[str]] = None
):
    plt.figure(figsize=(10, 8))
    
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    for i, cls in enumerate(unique_classes):
        mask = labels == cls
        label_name = class_names[cls] if class_names else f"Class {cls}"
        plt.scatter(reduced_features[mask, 0], reduced_features[mask, 1],
                   c=[colors[i]], label=label_name, alpha=0.6, s=20)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Component 1', fontsize=12)
    plt.ylabel('Component 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot to {save_path}")

def compute_class_separability_metrics(features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import silhouette_score
    
    unique_classes = np.unique(labels)
    
    centroids = []
    intra_class_vars = []
    
    for cls in unique_classes:
        mask = labels == cls
        class_features = features[mask]
        
        # Calculate centroid
        centroid = class_features.mean(axis=0)
        centroids.append(centroid)
        
        # Calculate intra-class variance
        var = np.mean(np.var(class_features, axis=0))
        intra_class_vars.append(var)
    
    centroids = np.array(centroids)
    
    # Inter-class variance (variance of centroids)
    inter_class_var = np.mean(np.var(centroids, axis=0))
    avg_intra_class_var = np.mean(intra_class_vars)
    separability_ratio = inter_class_var / (avg_intra_class_var + 1e-10)
    
    # Silhouette Score (subsample if large to avoid OOM/slow)
    if len(features) <= 10000:
        sil_score = silhouette_score(features, labels, sample_size=min(5000, len(features)))
    else:
        indices = np.random.choice(len(features), 5000, replace=False)
        sil_score = silhouette_score(features[indices], labels[indices])
    
    return {
        'intra_class_variance': float(avg_intra_class_var),
        'inter_class_variance': float(inter_class_var),
        'separability_ratio': float(separability_ratio),
        'silhouette_score': float(sil_score)
    }

def create_comparison_plot(
    all_reduced_features: Dict[str, np.ndarray], 
    labels: np.ndarray, 
    layer_names: List[str], 
    method: str, 
    save_path: Path, 
    class_names: Optional[List[str]] = None
):
    n_layers = len(layer_names)
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 5))
    
    if n_layers == 1:
        axes = [axes]
    
    unique_classes = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
    
    titles = ['Early', 'Middle', 'Late']
    
    for idx, (layer_name, ax) in enumerate(zip(layer_names, axes)):
        reduced = all_reduced_features[layer_name]
        
        for i, cls in enumerate(unique_classes):
            mask = labels == cls
            label_name = class_names[cls] if class_names else f"Class {cls}"
            # Only add legend to last plot
            ax.scatter(reduced[mask, 0], reduced[mask, 1],
                      c=[colors[i]], label=label_name if idx == n_layers - 1 else "",
                      alpha=0.6, s=15)
        
        stage_title = titles[idx] if idx < len(titles) else "Layer"
        ax.set_title(f"{layer_name}\n({stage_title} Layer)", fontsize=12, fontweight='bold')
        ax.set_xlabel('Component 1', fontsize=10)
        ax.set_ylabel('Component 2', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    if class_names:
        axes[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    plt.suptitle(f'Feature Hierarchy Visualization ({method.upper()})', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {save_path}")

def find_available_models(base_dir: str = "runs/task1/resnet") -> List[tuple]:
    models = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        return models
    
    # Recursively find .pt files
    for model_path in base_path.rglob("*.pt"):
        run_dir = model_path.parent
        
        # Heuristics to identify model type
        if 'modified' in str(model_path) or 'modified' in str(run_dir):
            model_type = 'modified'
        elif 'baseline' in str(run_dir) or 'head_only' in str(model_path):
            model_type = 'baseline'
        else:
            model_type = 'unknown'
            
        models.append((str(model_path), str(run_dir), model_type))
    
    return sorted(models, key=lambda x: x[1])

def display_available_models() -> Optional[List[tuple]]:
    print("\n" + "="*70)
    print("AVAILABLE MODEL CHECKPOINTS")
    print("="*70)
    
    models = find_available_models()
    
    if not models:
        print("\nNo model checkpoints found in runs/task1/resnet/")
        print("Please train a model first.")
        return None
    
    print(f"\nFound {len(models)} model checkpoint(s):\n")
    
    for idx, (model_path, run_dir, model_type) in enumerate(models, 1):
        model_name = Path(model_path).name
        run_name = Path(run_dir).name
        print(f"{idx}. [{model_type.upper()}] {model_name}")
        print(f"   Run: {run_name}")
        print(f"   Path: {model_path}")
        print()
    
    print("="*70)
    return models

def main():
    parser = argparse.ArgumentParser(description='Visualize feature hierarchies in ResNet')
    parser.add_argument('--model_path', type=str, required=False, help='Path to trained model weights')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for feature extraction')
    parser.add_argument('--max_batches', type=int, default=10, help='Maximum number of batches to process')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--methods', type=str, nargs='+', default=['tsne', 'umap'], help='DR methods')
    parser.add_argument('--exp', type=str, default='feature_hierarchies', help='Experiment name')
    args = parser.parse_args()
    
    if not args.model_path:
        display_available_models()
        return
    
    if not Path(args.model_path).exists():
        print(f"\nError: Model file not found: {args.model_path}")
        return
    
    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("Loading CIFAR-10 data...")
    _, val_loader = get_cifar10_loaders(batch_size=args.batch_size, num_workers=0)
    
    print("Loading model...")
    weights = ResNet152_Weights.DEFAULT
    model = resnet152(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    layer_groups = get_layer_groups('resnet152')
    layer_names = [layer_groups['early'], layer_groups['middle'], layer_groups['late']]
    
    print(f"Extracting features from layers: {layer_names}")
    
    extractor = FeatureExtractor(model, layer_names)
    
    features_dict, labels = extractor.extract_features(val_loader, device, max_batches=args.max_batches)
    extractor.remove_hooks()
    
    run_dir = make_run_dir("task1/resnet", args.exp)
    
    # Compute metrics
    print("\nCalculating metrics...")
    all_metrics = {}
    for name in layer_names:
        metrics = compute_class_separability_metrics(features_dict[name], labels)
        all_metrics[name] = metrics
        print(f"  {name}: Separability={metrics['separability_ratio']:.4f}")
    
    with open(run_dir / 'separability_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Plotting
    for method in args.methods:
        print(f"\nProcessing {method.upper()}...")
        all_reduced = {}
        for name in layer_names:
            reduced = apply_dimensionality_reduction(features_dict[name], labels, method=method, random_state=args.seed)
            all_reduced[name] = reduced
            
            stage = 'early' if name == layer_names[0] else ('middle' if name == layer_names[1] else 'late')
            plot_path = run_dir / f'{method}_{stage}_{name.replace(".", "_")}.png'
            plot_feature_space(reduced, labels, f'{method.upper()} - {name}', plot_path, CLASS_NAMES)
            
        create_comparison_plot(all_reduced, labels, layer_names, method, 
                              run_dir / f'{method}_comparison.png', CLASS_NAMES)
    
    print(f"\nAnalysis complete. Results: {run_dir}")

if __name__ == '__main__':
    main()
