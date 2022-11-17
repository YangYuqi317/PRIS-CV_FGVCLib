# Tutorial 6: Learn about model

In this part, we took the model apart and encapsulated each part of the model. We provide `backbones`, `encoders`, `heads`, `necks`, `sotas`, and `utils` for the model. You can select them separately to go to the component model.

- **Backbone**: The backbone network most of time refers to the feature extraction network, its role is to extract the information in the picture, and then use the network, e.g. **ResNet**, **VGG**.
- **Encoder**: The pooling layer can reduce the space size of the data body, so that the number of parameters in the network can be reduced, so that the computing resource consumption is reduced, and the overfitting can be effectively controlled.
- **Neck**: The component between backbones and heads.
- **Head**: The component for specific tasks.
- **Sotas**: State-of-the-art model.

## Backbone
We mainly provide two categories backbone, ResNet and VGG.
  
|              Backbone             |
| ResNet          | VGG             |
| resnet18        | vgg11           |
| resnet34        | vgg13           |
| resnet50        | vgg16           |
| resnet101       | vgg19           |
| resnet152       |                 |
| resnet50_32x4d  |                 |
| resnet101_32x8d |                 |
| resnet50_bc     |                 |
| resnet101_bc    |                 |


In the fgvclib/models/backbones/__init__.py", we define a function `get_backbone` to return the backbone with the givenname. The given names are `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`, `resnext50_32x4d`, `resnext101_32x8d`, `resnet50_bc`, `resnet101_bc`, `vgg11`, `vgg13`, `vgg16`, `vgg19`

```python
def get_backbone(backbone_name):
    if backbone_name not in globals():
        raise NotImplementedError(f"Backbone {backbone_name} not found!\nAvailable backbones: {__all__}")
    return globals()[backbone_name]
```

### ResNet
We download the ResNet-x model from pytorch, and define the functions to construct the ResNet-x model.

**resnet18:**
- Args:

   pretrained (bool): If True, returns a model pre-trained on ImageNet
   progress (bool): If True, displays a progress bar of the download to stderr

- Return:

   _resnet('resnet18', BasicBlock, [2, 2, 2, 2], cfg, progress, **kwargs)

This function return the `_resnet`, and `_resnet` return the model. The `_resnet` has some input parameters about the model category.

Other backbones are similar to the resnet18, the difference lies on the **return**.

**resnet50_32x4d**

- Return:
   
   _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], cfg, progress=True, **kwargs)

resnet50_32x4d needs to add the folllowing code:
```python
kwargs['groups'] = 32
kwargs['width_per_group'] = 4
```

**resnet101_32x8d**

- Return:
   _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], cfg, progress=True, **kwargs)

resnet101_32x8d needs to add the folllowing code:
```python
kwargs['groups'] = 32
kwargs['width_per_group'] = 8
```

### VGG
We download the VGG-x model from pytorch, and define the functions to construct the VGG-x model.

**vgg11:**
- Args:

   pretrained (bool): If True, returns a model pre-trained on ImageNet
   progress (bool): If True, displays a progress bar of the download to stderr

- Return:

   _vgg("vgg11", cfg, progress)

This function return the `_vgg`, and `_vgg` return the model. The `_vgg` has some input parameters about the model category.

Other backbones are similar to the vgg11, the difference lies on the **return**.

### The example
When you need to build a FGVC model, you can use it to get a backbone.
In the FGVCLib, we build a FGVC model according to config. For detailes about the **configs** , see [FGVC Configs](https://torchmetrics.readthedocs.io/en/stable/classification/recall.html).

In `fgvclib/apis/build.py`, there is a function build_model to build a FGVC model according to config. In the `model_cfg`, we have set the backbone name.
```python
from fgvclib.models.backbones import get_backbone

backbone_builder = get_backbone(model_cfg.BACKBONE.NAME)
backbone = backbone_builder(cfg=tltd(model_cfg.BACKBONE.ARGS))

```

## Encoders
We provide three kind of pooling layer, `global average pooling`, `global max pooling` and `max pooling 2d`.

In the fgvclib/models/encoders/__init__.py", we define a function `get_encoding` to return the encoder with the given name. And the given names are `global_avg_pooling`, `global_max_pooling`, `max_pooling_2d`

```python
def get_encoding(encoding_name):
    if encoding_name not in globals():
        raise NotImplementedError(f"Encoding not found: {encoding_name}\nAvailable encodings: {__all__}")
    return globals()[encoding_name]
```
### Global average pooling
Firstly, we define a class named `GlobalAvgPooling` as global average pooling encoding.
Then, we define a function named `global_avg_pooling`.

### Global max pooling
Firstly, we define a class named `GlobalMaxPooling` as global average pooling encoding.
Then, we define a function named `global_max_pooling`.

### Max pooling 2d

```python
def max_pooling_2d(cfg):
    assert 'kernel_size' in cfg.keys()
    assert isinstance(cfg['kernel_size'], int) 
    assert 'stride' in cfg.keys()
    assert isinstance(cfg['stride'], int)
    return nn.MaxPool2d(kernel_size=cfg['kernel_size'], stride=cfg['stride'])
```
### The example 
When you need to build a FGVC model, you can use it to get a encoding.
In the FGVCLib, we build a FGVC model according to config. For detailes about the **configs** , see [FGVC Configs](https://torchmetrics.readthedocs.io/en/stable/classification/recall.html).

In `fgvclib/apis/build.py`, there is a function build_model to build a FGVC model according to config. In the `model_cfg`, we have set the encoding name.
```python
from fgvclib.models.encoders import get_encoding

if model_cfg.ENCODING.NAME:
        encoding_builder = get_encoding(model_cfg.ENCODING.NAME)
        encoding = encoding_builder(cfg=tltd(model_cfg.ENCODING.ARGS))
    else:
        encoding = None

```

## Necks
We provide one kind neck for the fgvclib, `Multi-scale Convolution neck`.

In the fgvclib/models/necks/__init__.py", we define a function `get_neck` to return the neck with the given name. And the given name is `multi_scale_conv`.

```python
def get_neck(neck_name):
    """Return the backbone with the given name."""
    if neck_name not in globals():
        raise NotImplementedError(f"Neck not found: {neck_name}\nAvailable necks: {__all__}")
    return globals()[neck_name]
```

### Multi-scale Convolution neck
Firstly, we define a class named `MultiScaleConv` as a Multi-scale Convolution neck.
Then, we define a function named `multi_scale_conv`.

### The example
When you need to build a FGVC model, you can use it to get a neck.
In the FGVCLib, we build a FGVC model according to config. For detailes about the **configs** , see [FGVC Configs](https://torchmetrics.readthedocs.io/en/stable/classification/recall.html).

In `fgvclib/apis/build.py`, there is a function build_model to build a FGVC model according to config. In the `model_cfg`, we have set the neck name.
```python
from fgvclib.models.necks import get_neck

if model_cfg.NECKS.NAME:
    neck_builder = get_neck(model_cfg.NECKS.NAME)
    necks = neck_builder(cfg=tltd(model_cfg.NECKS.ARGS))
else:
    necks = None

```

## Heads
We mainly provide two classifier, `classifier_1fc`, and `classifier_2fc`

In the fgvclib/models/heads/__init__.py", we define a function `get_head` to return the head with the given name. And the given names are `classifier_1fc`, and `classifier_2fc`.

```python
def get_head(head_name):
    """Return the backbone with the given name."""
    if head_name not in globals():
        raise NotImplementedError(f"Head not found: {head_name}\nAvailable heads: {__all__}")
    return globals()[head_name]
```

### Classifier_1FC
Firstly, we define a class named `Classifier_1FC` as a classifier with one fully connected layer.
Then, we define a function named `classifier_1fc`.

### Classifier_2FC
Firstly, we define a class named `Classifier_2FC` as a classifier with two fully connected layer.
Then, we define a function named `classifier_2fc`.

### The example
When you need to build a FGVC model, you can use it to get a head.
In the FGVCLib, we build a FGVC model according to config. For detailes about the **configs** , see [FGVC Configs](https://torchmetrics.readthedocs.io/en/stable/classification/recall.html).

In `fgvclib/apis/build.py`, there is a function build_model to build a FGVC model according to config. In the `model_cfg`, we have set the head name.
```python
from fgvclib.models.heads import get_head

head_builder = get_head(model_cfg.HEADS.NAME)
    heads = head_builder(class_num=model_cfg.CLASS_NUM, cfg=tltd(model_cfg.HEADS.ARGS))
```

## Sotas
We reproduced state-of-the-art models, `baseline_resnet50`, `mcl`, `pmg_resnet50`, `pmg_v2_resnet50`.

In the fgvclib/models/sotas/__init__.py", we define a function `get_model` to return the head with the given name. And the given names are `PMG_ResNet50`, `PMG_V2_ResNet50`, `Baseline_ResNet50`, `MCL`.

```python
def get_model(model_name):
    """Return the model class with the given name."""
    if model_name not in globals():
        raise NotImplementedError(f"Model {model_name} not found!\nAvailable models: {__all__}")
    return globals()[model_name]
```

- Baseline_resnet50: Using the resnet50 as the backbone to build a model which is the baseline.
- MCL: This model was proposed in the paper "The Devil is in the Channels: Mutual-Channel Loss for Fine-Grained Image Classification", for more details about this method, see[MCL](https://arxiv.org/abs/2002.04264)
- PMG: This model was proposed in the paper " Fine-Grained Visual Classiﬁcation via Progressive Multi-Granularity Training of Jigsaw Patches", for more details about this method, see [PMG](https://arxiv.org/abs/2003.03836v3)

### The example
When you need to build a FGVC model, you can use it to get a model.
In the FGVCLib, we build a FGVC model according to config. For detailes about the **configs** , see [FGVC Configs](https://torchmetrics.readthedocs.io/en/stable/classification/recall.html).

In `fgvclib/apis/build.py`, there is a function build_model to build a FGVC model according to config. In the `model_cfg`, we have set the model name.

```python
from fgvclib.models.sotas import get_model

model_builder = get_model(model_cfg.NAME)
model = model_builder(backbone=backbone, encoding=encoding, necks=necks, heads=heads, criterions=criterions)
```

## Build a model
A model is made up **backbone**, **encoder**, **neck**, **head**, and **loss**. We take the model apart and you can combine them to build a new model or replicate other network. You should configure the model parameters in the configs in advance and then you can invoking these moudles to build model.

### Take an example to show the process of building a model.
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
