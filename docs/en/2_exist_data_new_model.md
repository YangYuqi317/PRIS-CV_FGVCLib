# 2: Train with the new models and satndard datastes
We disassemble the model into backbone, encoder, classifier and other basic structures, and then combine them to build the complete method. In the FGVCLib, we have provide the basic structures and reproduce the state-of-art model. We are committed to providing you with a custom structure, using the disassembled modules to reassemble a new model.

The basic steps are as below:
    1.Prepare the dataset
    2.Prepare you own customized model
    3.Prepare a config
    4.Train, test and predict models on the dataset

## Prepare the dataset
You need to change the corresponding dataset paths in the config files. And you need to split the dataset into train and test folder.

e.g., CUB-200-2011 dataset
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

## Prepare your own customized model
The second step is to use your own module or training setting. Assume that we want to add a new encoder `xxx`.

### 1.Define a new encoder(e.g. xxx)
Firstly we create a new file `fgvclib/model/encoders/xxx.py`.

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

### 2.Import the module
You can import the encoder in other parts
```python
from .xxx import xxx
```

## Prepare a config
The third step is to prepare a config for your own training setting. In "configs/xxx.yml", you should prepare a complete config file.
Refer to the existing config file, you can create a new config file.
Take the new encoder xxx as an example:
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

## Train a new model
To train a model with the new config, you can simply run
```python
python main.py --configs/xxx.yml --task train
```

## Test a new model
To train a model with the new config, you can simply run
```python
python main.py --configs/xxx.yml --task predict
```