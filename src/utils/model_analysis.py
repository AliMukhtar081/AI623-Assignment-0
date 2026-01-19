import torch
import torch.nn as nn
import numpy as np

def get_model_summary(model: nn.Module, input_size: tuple = (1, 3, 224, 224), device: str = "cpu"):
    original_device = next(model.parameters()).device
    model.eval()
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size_mb = (param_size + buffer_size) / (1024 ** 2)

    flops = 0
    
    def conv_flops_counter_hook(module, input, output):
        nonlocal flops
        batch_size = input[0].shape[0]
        output_dims = list(output.shape[2:])
        
        kernel_dims = list(module.kernel_size)
        in_channels = module.in_channels
        out_channels = module.out_channels
        groups = module.groups
        
        filters_per_channel = out_channels // groups
        conv_per_position_flops = int(np.prod(kernel_dims)) * in_channels * filters_per_channel
        
        active_elements_count = batch_size * int(np.prod(output_dims))
        
        overall_conv_flops = conv_per_position_flops * active_elements_count
        
        bias_flops = 0
        if module.bias is not None:
            bias_flops = out_channels * active_elements_count
        
        flops += overall_conv_flops + bias_flops

    def linear_flops_counter_hook(module, input, output):
        nonlocal flops
        batch_size = input[0].shape[0]
        input_features = input[0].shape[1]
        output_features = module.out_features
        
        mul_flops = input_features * output_features * batch_size
        add_flops = input_features * output_features * batch_size
        
        bias_flops = 0
        if module.bias is not None:
            bias_flops = output_features * batch_size
            
        flops += mul_flops + add_flops + bias_flops

    def bn_flops_counter_hook(module, input, output):
        nonlocal flops
        input_element_count = np.prod(input[0].shape)
        flops += 4 * input_element_count
        
    def relu_flops_counter_hook(module, input, output):
        nonlocal flops
        input_element_count = np.prod(input[0].shape)
        flops += input_element_count

    hooks = []
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            hooks.append(layer.register_forward_hook(conv_flops_counter_hook))
        elif isinstance(layer, nn.Linear):
            hooks.append(layer.register_forward_hook(linear_flops_counter_hook))
        elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            hooks.append(layer.register_forward_hook(bn_flops_counter_hook))
        elif isinstance(layer, nn.ReLU):
            hooks.append(layer.register_forward_hook(relu_flops_counter_hook))
            
    try:
        dummy_input = torch.randn(input_size).to(device)
        with torch.no_grad():
            model(dummy_input)
    except Exception as e:
        print(f"Warning: FLOP estimation failed: {e}")
        flops = -1

    for h in hooks:
        h.remove()
        
    model.to(original_device)

    return {
        "Total Parameters": total_params,
        "Trainable Parameters": trainable_params,
        "Non-Trainable Parameters": non_trainable_params,
        "Model Size (MB)": total_size_mb,
        "Estimated FLOPs": flops
    }

def print_model_analysis(model: nn.Module, input_size: tuple = (1, 3, 224, 224), run_name: str = "Model Analysis"):
    stats = get_model_summary(model, input_size)
    print(f"\n================ {run_name} ================")
    print(f"Total Parameters        : {stats['Total Parameters']:15,}")
    print(f"Trainable Parameters    : {stats['Trainable Parameters']:15,}")
    print(f"Non-Trainable Parameters: {stats['Non-Trainable Parameters']:15,}")
    print(f"Approx. Memory Size     : {stats['Model Size (MB)']:15.2f} MB")
    if stats['Estimated FLOPs'] != -1:
        print(f"Estimated FLOPs         : {stats['Estimated FLOPs']:15,.0f}")
    else:
        print(f"Estimated FLOPs         : {'N/A'}")
    print("==================================================\n")
