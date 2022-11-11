# Tutorial 3: Learn about criterions

In this folder "fgvclib/criterions" we provide different loss functions for the fgvclib.

We provide four loss functions, `cross_entropy_loss`, `binary_cross_entropy_loss`, `mean_square_error_loss` and `mutual_channel_loss`

| Loss functions         | Name                       |
| ---------------------- | -------------------------- |
| cross entropy loss     | cross_entropy_loss         |
| binary entropy loss    | binary_cross_entropy_loss  |
| mean square error loss | mean_square_error_loss     |
| mutual channel loss    | mutual_channel_loss        |

## Base loss function

`cross_entropy_loss`, `binary_cross_entropy_loss`, `mean_square_error_loss` are the base loss functions, and they are from PyTorch.
"fgvclib/criterions/base_loss.py": provides the base loss functions.

**cross_entropy_loss**: Build the cross entropy loss function.
- Args:

  ```cfg (CfgNode)```: The root node of config.

- Return:

  ```nn.Module```: The loss function.

**binary_cross_entropy_loss**:Build the binary cross entropy loss function.
- Args:

  ```cfg (CfgNode)```: The root node of config.

- Return:

  ```nn.Module```: The loss function.

**mean_square_error_loss**: Build the mean square error loss function.
- Args:

  ```cfg (CfgNode)```: The root node of config.

- Return:

  ```nn.Module```: The loss function.

## Mutual channel loss

"fgvclib/criterions/mutual_channel_loss.py": provides the mutual channel loss function which was proposed on "The Devil is in the Channels: Mutual-Channel Loss for Fine-Grained Image Classification".

```class MutualChannelLoss```: The mutual channel loss function.

- Args:

  ```height (int)```: The kernel size of average pooling.
  ```cnum (int)```: Channel numbers per class.
  ```div_weight (float)```: The weight for diversity part loss.
  ```dis_weight (float)```: The weight for discriminality part loss.

## Utils
In the "fgvclib/criterions/utils.py", we design a class named `LossItem`, and two functions, `compute_loss_value` and `detach_loss_value`.

**LossItem**: A dataclass object for store training loss
- Args:

  name (string): The loss item name.
  value (torch.Tensor): The value of loss.
  weight (float, optional): The weight of current loss item, default is 1.0.

**compute_loss_value**: A dataclass object for store training loss
- Args:

  items (List[LossItem]): The loss items.

- Return:
  
  Tensor: The total loss value.

**detach_loss_value**: Detach loss value from GPU.
- Args:

  items (List[LossItem]): The loss items.

- Return:
  
  Dict: A loss information dict whose key is loss name, value is loss value.

## The use of the criterions

### Build loss functions for training.
In the "fgvclib/apis/build.py", use the "fgvclib.criterions" to build loss functions for training. You can choose the loss function name `criterion_cfg['name']` from  `cross_entropy_loss`, `cross_entropy_loss`, `mean_square_error_loss` and `mutual_channel_loss`.
```python
from fgvclib.criterions import get_criterion

def build_criterion(criterion_cfg: CfgNode) -> nn.Module:
    criterion_builder = get_criterion(criterion_cfg['name'])
    criterion = criterion_builder(cfg=tltd(criterion_cfg['args']))
    return criterion
```

### Calculate loss functions.
Following is about how to calculate the loss, and you can replace the loss functions.
```python
from fgvclib.criterions.utils import LossItem

losses = list()
losses.append(LossItem(name='cross_entropy_loss', value=self.criterions['cross_entropy_loss']['fn'](x, targets)))
```

### Define the forward.
Set the ResNet50 for example.
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



