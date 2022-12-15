# Tutorial 6: Learn about model

在这一部分，我们将模型进行拆分，并对其进行封装。我们为模型提供了 `backbones`, `encoders`, `heads`, `necks`, `sotas`, 和 `utils`这些组件，你可以分别选择它们组装模型。

- **Backbone**: 骨干网大多书的时候指特征提取网络，它的作用是提取图片中的信息，然后使用该网络, 常见的有： **ResNet**, **VGG**等.
- **Encoder**: 池化层可以减小数据体的空间大小，从而减少网络中的参数数量，进而减少计算资源消耗，有效控制过拟合。
- **Neck**: backbone 和 head 之间的组成部分
- **Head**: 特定任务的组件
- **Sotas**: 最先进的模型

## Backbone

我们主要提供两类backbone， ResNet 和 VGG.

| ResNet          | VGG             |
| resnet18        | vgg11           |
| resnet34        | vgg13           |
| resnet50        | vgg16           |
| resnet101       | vgg19           |
| resnet152       |—————————————————|
| resnet50_32x4d  |—————————————————|
| resnet101_32x8d |—————————————————|
| resnet50_bc     |—————————————————|
| resnet101_bc    |—————————————————|

在"fgvclib/models/backbones/__init__.py"中，我们定义了`get_backbone`函数，根据给出的backbone名称返回对应的backbone。backbone名称如下：`resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`, `resnext50_32x4d`, `resnext101_32x8d`, `resnet50_bc`, `resnet101_bc`, `vgg11`, `vgg13`, `vgg16`, `vgg19`

```python
def get_backbone(backbone_name):
    if backbone_name not in globals():
        raise NotImplementedError(f"Backbone {backbone_name} not found!\nAvailable backbones: {__all__}")
    return globals()[backbone_name]
```

### ResNet

我们从Pytorch中加载了ResNet-x模型，并且定义了函数用于构造ResNet-x模型。

**resnet18:**
- 参数:

   pretrained (bool): 如果该值为True，则返回在ImageNet上的预训练模型
   progress (bool): 如果该值为True，则显示下载的进度条

- 返回值:

   _resnet('resnet18', BasicBlock, [2, 2, 2, 2], cfg, progress, **kwargs)


该函数返回`_resnet`，`_resnet`返回对应的模型，`_resnet`中包含关于模型类别的输入参数

其他的backbone和resnet18类似，不同的地方在于返回值

**resnet50_32x4d**

- 返回值:
   
   _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], cfg, progress=True, **kwargs)

resnet50_32x4d needs to add the folllowing code:
```python
kwargs['groups'] = 32
kwargs['width_per_group'] = 4
```

**resnet101_32x8d**

- 返回值:
   _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], cfg, progress=True, **kwargs)

resnet101_32x8d needs to add the folllowing code:
```python
kwargs['groups'] = 32
kwargs['width_per_group'] = 8
```

### VGG

我们从Pytorch中加载了VGG-x模型，并且定义了函数用于构造VGG-x模型。

**vgg11:**
- 参数:

   pretrained (bool): If True, returns a model pre-trained on ImageNet
   progress (bool): If True, displays a progress bar of the download to stderr

- 返回值:

   _vgg("vgg11", cfg, progress)

该函数返回`_vgg`，`_vgg`返回对应的模型，`_vgg`中包含关于模型类别的输入参数

其他的backbone和vgg11类似，不同的地方在于返回值

### 举例

当你需要建立一个FGVC模型，你可以使用它得到一个骨干网。
在FGVCLib，我们根据配置构建FGVC模型，有关配置**configs**的更多细节，请参考[FGVC Configs](https://torchmetrics.readthedocs.io/en/stable/classification/recall.html).

在`fgvclib/apis/build.py`，函数`build_model`根据配置构建FGVC模型，在`model_cfg`中，我们提前设置了backbone名称。

```python
from fgvclib.models.backbones import get_backbone

backbone_builder = get_backbone(model_cfg.BACKBONE.NAME)
backbone = backbone_builder(cfg=tltd(model_cfg.BACKBONE.ARGS))
```

## Encoders

我们提供了三种类型的池化层`global average pooling`, `global max pooling`和`max pooling 2d`

在"fgvclib/models/encoders/__init__.py"中，我们定义了`get_encoding`函数，根据提供的池化层类型返回对应的编码器。给出的池化层名称有：`global_avg_pooling`, `global_max_pooling`, `max_pooling_2d`

```python
def get_encoding(encoding_name):
    if encoding_name not in globals():
        raise NotImplementedError(f"Encoding not found: {encoding_name}\nAvailable encodings: {__all__}")
    return globals()[encoding_name]
```
### 全局平均池化

首先我们定义了一个类：`GlobalAvgPooling`作为全局平均池化编码器。然后我们定义了一个函数`global_avg_pooling`

### 全局最大池化

首先，我们定义了一个类：`GlobalMaxPooling`作为全局最大池化编码器。然后，我们定义了一个函数`global_max_pooling`

### Max pooling 2d

```python
def max_pooling_2d(cfg):
    assert 'kernel_size' in cfg.keys()
    assert isinstance(cfg['kernel_size'], int) 
    assert 'stride' in cfg.keys()
    assert isinstance(cfg['stride'], int)
    return nn.MaxPool2d(kernel_size=cfg['kernel_size'], stride=cfg['stride'])
```
### 举例

当你需要构建一个FGVC模型时，你可以使用它构建一个编码器。在FGVCLib中，我们根据配置构建FGVC模型，关于配置**configs**的细节，请参考[FGVC Configs](https://torchmetrics.readthedocs.io/en/stable/classification/recall.html).

在`fgvclib/apis/build.py`中，函数`build_model`根据配置构建FGVC模型，在`model_cfg`中，我们提前设置了编码器名称。
```python
from fgvclib.models.encoders import get_encoding

if model_cfg.ENCODING.NAME:
        encoding_builder = get_encoding(model_cfg.ENCODING.NAME)
        encoding = encoding_builder(cfg=tltd(model_cfg.ENCODING.ARGS))
    else:
        encoding = None

```

## Necks

我们为fgvclib提供了一种neck，`Multi-scale Convolution neck`，在"fgvclib/models/necks/__init__.py"中，我们定义了一个函数`get_neck`，根据给出的neck名称返回对应的neck。给出的neck名称有：`multi_scale_conv`

```python
def get_neck(neck_name):
    """Return the backbone with the given name."""
    if neck_name not in globals():
        raise NotImplementedError(f"Neck not found: {neck_name}\nAvailable necks: {__all__}")
    return globals()[neck_name]
```

### Multi-scale Convolution neck

首先，我们定义一个类`MultiScaleConv`作为Multi-scale Convolution neck，然后，我们定义了一个函数`multi_scale_conv`。

### 举例

当你需要构建一个FGVC模型时，你可以使用它构建一个neck。在FGVCLib中，我们根据配置构建FGVC模型，关于配置**configs**的细节，请参考[FGVC Configs](https://torchmetrics.readthedocs.io/en/stable/classification/recall.html).

在`fgvclib/apis/build.py`中，函数`build_model`根据配置构建FGVC模型，在`model_cfg`中，我们提前设置了neck名称。

```python
from fgvclib.models.necks import get_neck

if model_cfg.NECKS.NAME:
    neck_builder = get_neck(model_cfg.NECKS.NAME)
    necks = neck_builder(cfg=tltd(model_cfg.NECKS.ARGS))
else:
    necks = None

```

## Heads

我们主要提供两种分类器，`classifier_1fc`, and `classifier_2fc`，在"fgvclib/models/heads/__init__.py"中，我们定义了一个函数`get_head`，根据给出的head名称返回对应的head。给出的head名称有：`classifier_1fc`, and `classifier_2fc`

```python
def get_head(head_name):
    """Return the backbone with the given name."""
    if head_name not in globals():
        raise NotImplementedError(f"Head not found: {head_name}\nAvailable heads: {__all__}")
    return globals()[head_name]
```

### Classifier_1FC

首先，我们定义一个类：`Classifier_1FrC`作为具有一个全连接层的分类器，然后，我们定义一个函数`classifier_1fc`

### Classifier_2FC

首先，我们定义一个类：`Classifier_1FrC`作为具有两个全连接层的分类器，然后，我们定义一个函数`classifier_1fc`

### 举例

当你需要构建一个FGVC模型时，你可以使用它构建一个head。在FGVCLib中，我们根据配置构建FGVC模型，关于配置**configs**的细节，请参考[FGVC Configs](https://torchmetrics.readthedocs.io/en/stable/classification/recall.html).

在`fgvclib/apis/build.py`中，函数`build_model`根据配置构建FGVC模型，在`model_cfg`中，我们提前设置了head名称。

```python
from fgvclib.models.heads import get_head

head_builder = get_head(model_cfg.HEADS.NAME)
    heads = head_builder(class_num=model_cfg.CLASS_NUM, cfg=tltd(model_cfg.HEADS.ARGS))
```

## Sotas

我们复现了几个最先进的模型，`baseline_resnet50`, `mcl`, `pmg_resnet50`, `pmg_v2_resnet50`，在"fgvclib/models/heads/__init__.py"中，我们定义了一个函数`get_model`，根据给出的model名称返回对应的model。给出的model名称有：`PMG_ResNet50`, `PMG_V2_ResNet50`, `Baseline_ResNet50`, `MCL`

```python
def get_model(model_name):
    """Return the model class with the given name."""
    if model_name not in globals():
        raise NotImplementedError(f"Model {model_name} not found!\nAvailable models: {__all__}")
    return globals()[model_name]
```

- Baseline_resnet50: 使用resnet50作为主干网络去构建模型作为基准模型
- MCL: 这个模型在"The Devil is in the Channels: Mutual-Channel Loss for Fine-Grained Image Classification"论文中被提出，关于此模型的更多细节参考[MCL](https://arxiv.org/abs/2002.04264)
- PMG: 这个模型在" Fine-Grained Visual Classiﬁcation via Progressive Multi-Granularity Training of Jigsaw Patches"论文中被提出, 关于此模型的更多细节参考[PMG](https://arxiv.org/abs/2003.03836v3)

### 举例

当你需要构建一个FGVC模型时，你可以使用它获得模型，在FGVCLib中，我们根据配置构建FGVC模型，关于配置**configs**的更多细节，请参考[FGVC Configs](https://torchmetrics.readthedocs.io/en/stable/classification/recall.html).

在`fgvclib/apis/build.py`中，函数`build_model`根据配置构建FGVC模型，在`model_cfg`中，我们提前设置了model名称

```python
from fgvclib.models.sotas import get_model

model_builder = get_model(model_cfg.NAME)
model = model_builder(backbone=backbone, encoding=encoding, necks=necks, heads=heads, criterions=criterions)
```

## 构建一个模型

一个完整的模型由**backbone**, **encoder**, **neck**, **head**, 和 **loss**这几部分组成，我们将模型的各个部分进行拆分，你可以自由的组合它们去构建一个新的模型，或者复现其他的工作，你需要事先在配置中设置好模型的参数，才能调用这些模块来构建模型。

### 举例说明构建模型的过程
```python
from fgvclib.metrics import get_metric
from fgvclib.models.sotas import get_model
from fgvclib.models.backbones import get_backbone
from fgvclib.models.encoders import get_encoding
from fgvclib.models.necks import get_neck
from fgvclib.models.heads import get_head

def build_model(model_cfg: CfgNode) -> nn.Module:
    r"""Build a FGVC model according to config.

    Args:
        model_cfg (CfgNode): The model config node of root config.
    Returns:
        nn.Module: The FGVC model.
    """

    backbone_builder = get_backbone(model_cfg.BACKBONE.NAME)
    backbone = backbone_builder(cfg=tltd(model_cfg.BACKBONE.ARGS))

    if model_cfg.ENCODING.NAME:
        encoding_builder = get_encoding(model_cfg.ENCODING.NAME)
        encoding = encoding_builder(cfg=tltd(model_cfg.ENCODING.ARGS))
    else:
        encoding = None

    if model_cfg.NECKS.NAME:
        neck_builder = get_neck(model_cfg.NECKS.NAME)
        necks = neck_builder(cfg=tltd(model_cfg.NECKS.ARGS))
    else:
        necks = None

    head_builder = get_head(model_cfg.HEADS.NAME)
    heads = head_builder(class_num=model_cfg.CLASS_NUM, cfg=tltd(model_cfg.HEADS.ARGS))

    criterions = {}
    for item in model_cfg.CRITERIONS:
        criterions.update({item["name"]: {"fn": build_criterion(item), "w": item["w"]}})
    
    model_builder = get_model(model_cfg.NAME)
    model = model_builder(backbone=backbone, encoding=encoding, necks=necks, heads=heads, criterions=criterions)
    
    return model
```
