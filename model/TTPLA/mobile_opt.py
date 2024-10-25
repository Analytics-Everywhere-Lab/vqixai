from torch import nn
from torch.nn.utils import prune

from model.substation.config import *

if __name__ == "__main__":
    model = torch.load(MODEL_PATH, map_location=DEVICE)
    model.to(DEVICE)
    model.eval()

    initial_params = sum(p.numel() for p in model.parameters())
    initial_size_bytes = sum(p.element_size() * p.nelement() for p in model.parameters())
    print("Initial number of parameters:", initial_params)
    # Print size in MB
    initial_size_bytes /= 1024 * 1024
    print("Initial size of model (MB):", initial_size_bytes)

    # Disable batch normalization
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    # Apply quantization (dynamic as example)
    quantized_model = torch.quantization.quantize_dynamic(model,
                                                          {nn.Conv2d, nn.Linear, nn.ReLU, nn.BatchNorm2d,
                                                           nn.AdaptiveAvgPool2d},
                                                          dtype=torch.qint8)

    # Dummy forward pass to simulate inference
    dummy_input = torch.randn(1, 3, 224, 224)
    quantized_model(dummy_input)

    # Apply pruning
    for name, module in quantized_model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=0.1, n=1, dim=0)

    # Remove pruning reparameterizations
    for name, module in quantized_model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.remove(module, 'weight')

    # Calculate effective size considering zero values
    effective_size_bytes = sum((p != 0).sum().item() * p.element_size() for p in quantized_model.parameters())
    print("Pruned number of parameters (effective):", sum((p != 0).sum().item() for p in quantized_model.parameters()))
    # Print size in MB
    effective_size_bytes /= 1024 * 1024
    print("Pruned size of model (MB):", effective_size_bytes)
