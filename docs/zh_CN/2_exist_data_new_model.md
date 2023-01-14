# 2: 在标准数据集上训练自定义模型

我们将模型分解为骨干、编码器、分类器等基本结构，然后将它们组合起来构建完整的方法。在FGVCLib中，我们提供了基本结构并复现了最先进的模型。我们致力于为您提供自定义结构，使用分解后的模块重新组装成新的模型。

主要的步骤如下：
    1.准备数据集

    2.准备自定义模型

    3.准备配置文件
    
    4.在标准数据集上进行训练、测试和推理

## 准备数据集
你需要在配置文件中修改对应的数据集路径。你需要将数据集分成训练集和测试集两个文件夹。

例如，CUB-200-2011数据集：
```
  -/birds/train
	         └─── 001.Black_footed_Albatross
	                   └─── Black_Footed_Albatross_0001_796111.jpg
	                   └─── ...
	         └─── 002.Laysan_Albatross
	         └─── 003.Sooty_Albatross
	         └─── ...
   -/birds/test	
             └─── ...         
```

## 准备自定义模型
第二步时使用已有的模块和新的模块构建自定义模型，假设我们想添加一个新的编码器`xxx`。

### 1.定义一个新的编码器（以xxx为例）
首先建立新文件`fgvclib/model/encoders/xxx.py`。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from fgvclib.models.encoders import encoder

class xxx(nn.Module):
    def __init__(self)
        pass

    def forward(self,inputs):
        pass

def xxx(cfg:dict):
    pass
```

### 2.导入模块
你可以在其他需要的地方导入该编码器
```python
from .xxx import xxx
```

## 准备配置文件
第三步是为你自己的训练设置准备一个配置文件。在"configs/xxx.yml"中，你可以根据已有的配置文件，新建立配置。

以新编码器xxx为例：
```python
MODEL:
  NAME: "ResNet50"
  CLASS_NUM: 200
  CRITERIONS: 
    - name: "cross_entropy_loss"
      args: []
      w: 1.0
  BACKBONE:
    NAME: "resnet50"
    ARGS:
      - pretrained: True
      - del_keys: []
  ENCODER:
    NAME: "xxx"
  NECKS:
    NAME: ~
  HEADS:
    NAME: "classifier_1fc"
    ARGS:
      - in_dim: 
        - 2048
```

## 训练新模型
为了能够使用新增配置来训练模型，你可以运行如下命令：
```python
python main.py --configs/xxx.yml --task train
```

## 测试新模型
为了能够测试训练好的模型，你可以运行如下命令：
```python
python main.py --configs/xxx.yml --task predict
```