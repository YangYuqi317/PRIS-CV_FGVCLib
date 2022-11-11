# Benchmark and model zoo

## Common settings

- All models were trained on `CUB_200_2011_train` and tested on the `CUB_200_2011_test`.
- For fair comparison with other codebases, we report the GPU memory as the maximum value of `torch.cuda.max_memory_allocated()` for all 8 GPUs. Note that this value is usually less than what `nvidia-smi` shows.
- All pytorch-style pretrained backbone are form PyTorch model zoo.

## Backbone models

The detailed table of the commonly used backbone models in FGVCLib is listed below:

| model            | source      | link                                                                                                                                                                                                   | description                                                                                                                                                                                                                                      |
| ---------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ResNet18         | TorchVision | [torchvision's ResNet-18](https://download.pytorch.org/models/resnet18-5c106cde.pth)                                                                                                                   | From [torchvision's ResNet-18](https://download.pytorch.org/models/resnet18-5c106cde.pth).                                                                                                                                                       |
| ResNet34         | TorchVision | [torchvision's ResNet-34](https://download.pytorch.org/models/resnet34-333f7ec4.pth)                                                                                                                   | From [torchvision's ResNet-34](https://download.pytorch.org/models/resnet34-333f7ec4.pth).                                                                                                                                                       |
| ResNet50         | TorchVision | [torchvision's ResNet-50](https://download.pytorch.org/models/resnet50-19c8e357.pth)                                                                                                                   | From [torchvision's ResNet-50](https://download.pytorch.org/models/resnet50-19c8e357.pth).                                                                                                                                                       |
| ResNet101         | TorchVision | [torchvision's ResNet-101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)                                                                                                                   | From [torchvision's ResNet-101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth).                                                                                                                                                       |
| ResNet152         | TorchVision | [torchvision's ResNet-152](https://download.pytorch.org/models/resnet152-b121ed2d.pth)                                                                                                                   | From [torchvision's ResNet-152](https://download.pytorch.org/models/resnet152-b121ed2d.pth).                                                                                                                                                       |
| Vgg11         | TorchVision | [torchvision's Vgg-11](https://download.pytorch.org/models/vgg11-bbd30ac9.pth)                                                                                                                   | From [torchvision's Vgg-11](https://download.pytorch.org/models/vgg11-bbd30ac9.pth).                                                                                                                                                       |
| Vgg13         | TorchVision | [torchvision's Vgg-13](https://download.pytorch.org/models/vgg13-c768596a.pth)                                                                                                                   | From [torchvision's Vgg-13](https://download.pytorch.org/models/vgg13-c768596a.pth).                                                                                                                                                       |
| Vgg16         | TorchVision | [torchvision's Vgg-16](https://download.pytorch.org/models/vgg16-397923af.pth)                                                                                                                   | From [torchvision's Vgg-16](https://download.pytorch.org/models/vgg16-397923af.pth).                                                                                                                                                       |
| Vgg19         | TorchVision | [torchvision's Vgg-19](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth)                                                                                                                   | From [torchvision's Vgg-19](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth).                                                                                                                                                       |

## Methods

### MCL
Please refer to [MCL](https://arxiv.org/abs/2002.04264) for details.

### PMG 
Please refer to [PMG](https://arxiv.org/abs/2003.03836v3) for details.

