# Tutorial 1: Learn about apis

In this folder "fgvclib/api" we set up the various apis interfaces for the fgvclib.
There are 4 types apis interfaces in this folder, ```build.py```, ```evluate_model.py```, ```save_model.py```, and ```update_model.py```

"fgvclib/apis/build.py" provides various apis for building a training or evaluation system fast.

"fgvclib/apis/evaluate_model.py" provides a api for evaluating FGVC algorithms.

"fgvclib/apis/save_model.py"provides various apis for saving a model.

"fgvclib/apis/update_model"provides various apis for updating models and recording losses.


## Build
We import the other modules to build the model, in this part, there are eight functions. The configs about the model are saved in the folder "./configs". For more detailes about the **configs** , see [FGVC Configs](https://docs-yyq.readthedocs.io/en/latest/global_configs.html)

**build_model**: Build a FGVC model according to config.
- Args:

    `model_cfg (CfgNode)`: The model config node of root config.

- Returns:

    `nn.Module`: The FGVC model.

**build_logger**: Build a Logger object according to config.
- Args:

    `cfg (CfgNode)`: The root config node.

- Returns:

    `Logger`: The Logger object.

**build_transforms**: Build transforms for train or test dataset according to config.
- Args:

    `transforms_cfg (CfgNode)`: The root config node.

- Returns:

    `transforms.Compose`: The transforms.Compose object in Pytorch.

**build_dataset**: Build a dataloader for training or evaluation.
- Args: 

    `name(str)`: The dataset name.
    `root (str)`: The directory of dataset.
    `cfg (CfgNode)`: The mode config of the dataset config.
    `mode(str)`: The split of the dataset.
    `transform`: Pytorch Transformer Compose.

- Returns:

    `DataLoader`: A Pytorch Dataloader.

**build_optimizer**: Build a optimizer for training.
- Args:

    `optim_cfg (CfgNode)`: The optimizer config node of root config node.

- Returns:

    `Optimizer`: A Pytorch Optimizer.

**build_criterion** :Build loss function for training.
- Args:

   `criterion_cfg` (CfgNode): The criterion config node of root config node.

- Returns:

    `nn.Module`: A loss function.

**build_interpreter**: Build loss function for training.
- Args:

    `cfg (CfgNode)`: The root config node.

- Returns:

    `Interpreter`: A Interpreter.

**build_metrics**: Build metrics for evaluation.
- Args:

    `metrics_cfg (CfgNode)`: The metric config node of root config node.
    
- Returns:

    `t.List[NamedMetric]`: A List of NamedMetric.

## Evaluate Model

**build_model**: Build a FGVC model according to config.
- Args:

    `model_cfg (CfgNode)`: The model config node of root config.
- Returns:

    `nn.Module`: The FGVC model.

**build_logger**: Build a Logger object according to config.
- Args:

    `cfg (CfgNode)`: The root config node.
- Returns:

    `Logger`: The Logger object.

**build_transforms**: Build transforms for train or test dataset according to config.
- Args:

    `transforms_cfg (CfgNode)`: The root config node.
- Returns:

    `transforms.Compose`: The transforms.Compose object in Pytorch.

**build_dataset**: Build a dataloader for training or evaluation.
- Args:

    `root (str)`: The directory of dataset.
    `cfg (CfgNode)`: The root config node.
- Returns:
    `DataLoader`: A Pytorch Dataloader.

**build_optimizer**: Build a optimizer for training.
- Args:

    `optim_cfg (CfgNode)`: The optimizer config node of root config node.
- Returns:

    `Optimizer`: A Pytorch Optimizer.

**build_criterion**: Build loss function for training.
- Args:

    `criterion_cfg (CfgNode)`: The criterion config node of root config node.
- Returns:

    `nn.Module`: A loss function.

**build_interpreter**: Build loss function for training.
- Args:

    `cfg (CfgNode)`: The root config node.
- Returns:

    `Interpreter`: A Interpreter.

**build_metrics**: Build metrics for evaluation.
- Args:

    `metrics_cfg (CfgNode)`: The metric config node of root config node.
- Returns:

    `t.List[NamedMetric]`: A List of NamedMetric.

## Save Model

**save_model**: Save the trained FGVC model.
- Args:

    `cfg (CfgNode)`: The root config node.
    `model (nn.Module)`: The FGVC model.
    `logger (Logger)`: The Logger object.

## Update Model

**update_model**: Update the FGVC model and record losses.
- Args:

    `model (nn.Module)`: The FGVC model.
    `optimizer (Optimizer)`: The Logger object.
    `pbar (Iterable)`: A iterable object provide training data.
    `strategy (string)`: The update strategy.
    `use_cuda (boolean)`: Whether to use GPU to train the model.
    `logger (Logger)`: The Logger object.

## The use of the apis
When you do algorithm design, you need to import the apis ```from fgvclib.apis import * ```and call the interfaces. You can use the following functions directly, ```build_logger```,  ```build_criterion```, ```build_model```, ```build_metrics```, ```build_transforms```, ```build_dataset```, ```build_optimizer```, ```update_model```, ```evaluate_model```, ```save_model```, ```build_interpreter```

- The example of building model
```python
import os
import torch

from fgvclib.apis import *
from fgvclib.configs import FGVCConfig

model = build_model(cfg.MODEL)
weight_path = os.path.join(cfg.WEIGHT.SAVE_DIR, cfg.WEIGHT.NAME)
assert os.path.exists(weight_path), f"The resume weight {cfg.RESUME_WEIGHT} dosn't exists."
state_dict = torch.load(weight_path, map_location="cpu")
model.load_state_dict(state_dict=state_dict)

if cfg.USE_CUDA:
    assert torch.cuda.is_available(), f"Cuda is not available."
    model = torch.nn.DataParallel(model)

transforms = build_transforms(cfg.TRANSFORMS.TEST)
loader = build_dataset(root=os.path.join(cfg.DATASETS.ROOT, 'test'), cfg=cfg.DATASETS.TEST, transforms=transforms)

interpreter = build_interpreter(model, cfg)
voxel = VOXEL(dataset=loader.dataset, name=cfg.FIFTYONE.NAME, interpreter=interpreter)
voxel.predict(model, transforms, 10, cfg.MODEL.NAME)
voxel.launch()
```