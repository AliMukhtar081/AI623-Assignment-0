
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional

class FeatureExtractor:
    """
    Helper class to extract features from intermediate layers of a model using forward hooks.
    """
    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.model = model
        self.layer_names = layer_names
        self.features = {name: [] for name in layer_names}
        self.hooks = []
        
        for name in layer_names:
            layer = self._get_layer_by_name(name)
            hook = layer.register_forward_hook(self._make_hook(name))
            self.hooks.append(hook)
    
    def _get_layer_by_name(self, name: str) -> nn.Module:
        parts = name.split('.')
        layer = self.model
        for part in parts:
            layer = getattr(layer, part)
        return layer
    
    def _make_hook(self, name: str):
        def hook(module, input, output):
            # Clone to avoid reference issues, move to CPU immediately to save GPU memory
            self.features[name].append(output.detach().cpu())
        return hook
    
    def extract_features(
        self, 
        data_loader: torch.utils.data.DataLoader, 
        device: str, 
        max_batches: Optional[int] = None
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        
        self.model.eval()
        labels_list = []
        
        accumulated_features = {name: [] for name in self.layer_names}
        
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(data_loader, desc="Extracting features")):
                if max_batches is not None and i >= max_batches:
                    break
                    
                x = x.to(device)
                
                # Clear previous batch features
                for name in self.layer_names:
                    self.features[name].clear()
                
                _ = self.model(x)
                
                for name in self.layer_names:
                    # Get the feature from the hook
                    if not self.features[name]:
                        continue
                        
                    features = self.features[name][0]
                    
                    # Global Average Pooling for 4D tensors (B, C, H, W) -> (B, C)
                    if len(features.shape) == 4:
                        features = features.mean(dim=[2, 3])
                    
                    # Convert to numpy and store
                    features_np = features.float().numpy()
                    accumulated_features[name].append(features_np)
                
                labels_list.append(y.numpy())
        
        labels = np.concatenate(labels_list, axis=0)
        
        processed_features = {}
        for name in self.layer_names:
            if accumulated_features[name]:
                features = np.concatenate(accumulated_features[name], axis=0)
                processed_features[name] = features
            else:
                processed_features[name] = np.array([])
                
        return processed_features, labels
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()
