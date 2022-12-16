# 教程 3: 学习标准文件

在"fgvclib/criterions"这个文件夹下，我们为fgvclib提供了不同了损失函数。

我们提供了四个损失函数：`cross_entropy_loss`, `binary_cross_entropy_loss`, `mean_square_error_loss` 和 `mutual_channel_loss`

| Loss functions         | Name                       |
| ---------------------- | -------------------------- |
| 交叉熵损失              | cross_entropy_loss         |
| 二元交叉熵损失            | binary_cross_entropy_loss  |
| 均方差损失              | mean_square_error_loss     |
| 互信道损失              | mutual_channel_loss        |

## 基础的损失函数

`cross_entropy_loss`, `binary_cross_entropy_loss`, `mean_square_error_loss` 这三类损失函数是基础的损失函数，在fgvclib中，我们从Pytorch中调用它们。

"fgvclib/criterions/base_loss.py"中提供了这三类基础的损失函数。

**cross_entropy_loss**: 构建交叉熵损失函数
- 参数:

  ```cfg (CfgNode)```: 配置的根节点

- 返回值:

  ```nn.Module```: 损失函数

**binary_cross_entropy_loss**: 构建二元交叉熵损失函数
- 参数:

  ```cfg (CfgNode)```: 配置的根节点

- 返回值:

  ```nn.Module```: 损失函数

**mean_square_error_loss**: 构建均方差损失函数
- 参数:

  ```cfg (CfgNode)```: 配置的根节点

- 返回值:

  ```nn.Module```: 损失函数

## 互信道损失函数

"fgvclib/criterions/mutual_channel_loss.py" 提供了互信道损失函数，该方法在"The Devil is in the Channels: Mutual-Channel Loss for Fine-Grained Image Classification"论文中被提出，关于互信道损失函数的更多细节，参考该篇论文[MC-Loss](https://arxiv.org/abs/2002.04264)

```class MutualChannelLoss```: 互信道损失函数类
- 参数:

  ```height (int)```: average pooling 的内核大小
  ```cnum (int)```: 每个类的通道数量
  ```div_weight (float)```: 多样性部分损失的权重
  ```dis_weight (float)```: 判别性部分损失的权重

## 工具


在"fgvclib/criterions/utils.py"中，我们设计了一个类：`LossItem`，两个函数：`compute_loss_value` 和 `detach_loss_value`

**LossItem**: 用于储存训练损失的数据类对象
- 参数:

  `name (string)`: 损失函数名称
  `value (torch.Tensor)`: 损失项的值
  `weight (float, optional)`: 当前损失项的权重，默认为1.0

**compute_loss_value**: 用于储存训练损失的数据类对象
- 参数:

  `items (List[LossItem])`: 损失项

- 返回值:
  
  `Tensor`: 总的损失项的值

**detach_loss_value**: 从GPU分离损失值
- 参数:

  `items (List[LossItem])`: 损失项

- 返回值:
  
  `Dict`: 损失信息字典，key为损失名称，对应的值为损失值

## Criterion标准的应用

### 为训练过程建立损失函数

在"fgvclib/apis/build.py"中，使用"fgvclib.criterions"去为训练过程构建损失函数，你可以从这四类损失函数中选择`cross_entropy_loss`, `cross_entropy_loss`, `mean_square_error_loss` and `mutual_channel_loss`替换损失函数名称`criterion_cfg['name']`

```python
from fgvclib.criterions import get_criterion

def build_criterion(criterion_cfg: CfgNode) -> nn.Module:
    criterion_builder = get_criterion(criterion_cfg['name'])
    criterion = criterion_builder(cfg=tltd(criterion_cfg['args']))
    return criterion
```

### 计算损失函数

以下展示了如何计算损失，你可以替换其中的损失函数类型。

```python
from fgvclib.criterions.utils import LossItem

losses = list()
losses.append(LossItem(name='cross_entropy_loss', value=self.criterions['cross_entropy_loss']['fn'](x, targets)))
```

### 定义前向传播
以ResNet50结构为例：

```python
from fgvclib.criterions.utils import LossItem

def forward(self, x, targets=None):
    x = self.infer(x)
    if self.training:
        losses = list()
        osses.append(LossItem(name='cross_entropy_loss', value=self.criterions['cross_entropy_loss']['fn'](x, targets)))
        return x, losses
        
    return x
```



