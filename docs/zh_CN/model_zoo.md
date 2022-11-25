# Benchmark and model zoo

## 常规设置

- 所有的模型我们都是在`CUB_200_2011_train`上进行训练并且在`CUB_200_2011_test`上进行测试的
- 为了与其他代码库进行公平的比较，我们将GPU内存报告为所有8个GPU的' torch.cuda.max_memory_allocate() '的最大值。注意，这个值通常小于' nvidia-smi '显示的值。
- 所有的预训练的backbone都是来自于PyTorch model zoo。

## Backbone 模型

下面列出了FGVCLib中常用的骨干模型的详细表:

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

## 方法

### MCL
Please refer to [MCL](https://arxiv.org/abs/2002.04264) for details.

### PMG 
Please refer to [PMG](https://arxiv.org/abs/2003.03836v3) for details.

