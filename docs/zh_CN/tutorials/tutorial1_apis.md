# 教程 1: 学习接口文件

在"fgvclib/api"这个文件夹下，我们为fgvclib设置了各类api接口。这里有四种类型的api接口：```build.py```, ```evluate_model.py```, ```save_model.py```, 和 ```update_model.py```。

"fgvclib/apis/build.py"：提供了各种用于快速构建训练系统或评估系统的api；

"fgvc/apis/evluate_model.py"：提供了用于评估FGVC算法的api；

"fgvclib/apis/save_model.py"：提供了各种用于保存模型的api；

"fgvclib/apis/update_model"：提供了各种用于更新模型和记录损失的api。

## 模型构建

**build_model**: 根据全局配置构建一个FGVC模型。
- 参数:

    `model_cfg (CfgNode)`: 根配置的模型配置节点
- 返回值:

    `nn.Module`: FGVC模型

**build_logger**: 根据配置构建日志对象。
- 参数:

    `cfg (CfgNode)`: 根配置节点
- 返回值:

    `Logger`: 日志对象
**build_transforms**: 根据配置为训练或测试数据集构建转换
- 参数:

    `transforms_cfg (CfgNode)`: 根配置节点
- 返回值:

    `transforms.Compose`: Pytorch中的transforms.Compose对象

**build_dataset**: 为训练过程或评估过程构建数据加载器
- 参数: 

    `root (str)`: 数据集的目录
    `cfg (CfgNode)`: 根配置节点
- 返回值:

    `DataLoader`: Pytorch数据加载器

**build_optimizer**: 为训练过程构建优化器
- 参数:

    `optim_cfg (CfgNode)`: 根配置节点的优化配置节点
- 返回值:

    `Optimizer`: Pytorch优化器

**build_criterion** : 为训练过程构建损失函数
- 参数:

   `criterion_cfg` (CfgNode): 根配置节点的标准配置节点
- 返回值:

    `nn.Module`: 损失函数

**build_interpreter**: 为训练过程构建一个解释器
- 参数:

    `cfg (CfgNode)`: 根配置节点
- 返回值:

    `Interpreter`: 一个解释器

**build_metrics**: 为评估过程构建度量标准
- 参数:

    `metrics_cfg (CfgNode)`: 根配置节点的度量标准配置节点
- 返回值:

    `t.List[NamedMetric]`: NamedMetric列表

## 模型评估

**evaluate_model**:对FGVC模型进行评估
- 参数：

    `model (nn.Module)`: FGVC模型
    `p_bar (iterable)`: 提供测试数据的迭代器
    `metrics (List[NamedMetric])`: 指标的列表
    `use_cuda (boolean, optional)`: 是否使用gpu

- 返回值：

    `dict`: 结果的字典

## 模型保存

**save_model**: 保存被训练的FGVC模型
- 参数:

    `cfg (CfgNode)`: 根配置节点
    `model (nn.Module)`: FGVC模型
    `logger (Logger)`: 日志对象

## 模型更新

**update_model**: 更新FGVC模型并且记录损失
- 参数:

    `model (nn.Module)`: FGVC模型
    `optimizer (Optimizer)`: 日志对象
    `pbar (Iterable)`: 提供训练数据的可迭代对象
    `strategy (string)`: 更新的策略
    `use_cuda (boolean)`: 是否使用GPU训练模型
    `logger (Logger)`: 日志对象

## API的应用
当你进行算法设计时，你需要使用```from fgvclib.apis import * ```导入上述这些api去调用这些接口。你可以直接使用以下的函数：```build_logger```,  ```build_criterion```, ```build_model```, ```build_metrics```, ```build_transforms```, ```build_dataset```, ```build_optimizer```, ```update_model```, ```evaluate_model```, ```save_model```, ```build_interpreter```

- 应用举例：建立模型
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