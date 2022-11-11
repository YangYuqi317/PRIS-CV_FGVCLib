# Tutorial 2: Learn about configs

In this folder "fgvclib/configs" we show the configs about the fgvclib.
We modularized the config in the experiment, creating ```FGVCConfig``` class to load and store the parameters.And you can load config by using ```FGVCConfig```

## Config File Structure

There are 4 basic component types under "fgvclib/configs/config.py",```__init__```,```get_cfg```,```load```,```stringfy```.

We set the parameters for the fgvclib, and you can search or modify the parameters in ```config.py```.

The following is about the basic parameters.
```python
  # Name of Project
  self.cfg.PROJ_NAME = "FGVC"

  # Name of experiment
  self.cfg.EXP_NAME = None

  # Resume last train
  self.cfg.RESUME_WEIGHT = None

  # Directory of trained weight
  self.cfg.WEIGHT = CN()
  self.cfg.WEIGHT.NAME = None
  self.cfg.WEIGHT.SAVE_DIR = "./checkpoints/"

  # Use cuda
  self.cfg.USE_CUDA = True

  # Logger
  self.cfg.LOGGER = CN()
  self.cfg.LOGGER.NAME = "wandb_logger"
  self.cfg.LOGGER.FILE_PATH = "./logs/"
  self.cfg.LOGGER.PRINT_FRE = 50
```

The following parameters are about datasets.
```python
  # Datasets and data loader
  self.cfg.DATASET = CN()
  self.cfg.DATASET.NAME = None
  self.cfg.DATASET.ROOT = None
  self.cfg.DATASET.TRAIN = CN()
  self.cfg.DATASET.TEST = CN()

  # train dataset and data loder
  self.cfg.DATASET.TRAIN.BATCH_SIZE = 32
  self.cfg.DATASET.TRAIN.POSITIVE = 0
  self.cfg.DATASET.TRAIN.PIN_MEMORY = True
  self.cfg.DATASET.TRAIN.SHUFFLE = True
  self.cfg.DATASET.TRAIN.NUM_WORKERS = 0
        
  # test dataset and data loder
  self.cfg.DATASET.TEST.BATCH_SIZE = 32
  self.cfg.DATASET.TEST.POSITIVE = 0
  self.cfg.DATASET.TEST.PIN_MEMORY = False
  self.cfg.DATASET.TEST.SHUFFLE = False
  self.cfg.DATASET.TEST.NUM_WORKERS = 0
```

The following parameters are about the model.
```python
  # Model architecture
  self.cfg.MODEL = CN()
  self.cfg.MODEL.NAME = None
  self.cfg.MODEL.CLASS_NUM = None
  self.cfg.MODEL.CRITERIONS = None

  # Standard modulars of each model
  self.cfg.MODEL.BACKBONE = CN()
  self.cfg.MODEL.ENCODING = CN()
  self.cfg.MODEL.NECKS = CN()
  self.cfg.MODEL.HEADS = CN()
        
  # Setting of backbone
  self.cfg.MODEL.BACKBONE.NAME = None
  self.cfg.MODEL.BACKBONE.ARGS = None

  # Setting of encoding
  self.cfg.MODEL.ENCODING.NAME = None
  self.cfg.MODEL.ENCODING.ARGS = None

  # Setting of neck
  self.cfg.MODEL.NECKS.NAME = None
  self.cfg.MODEL.NECKS.ARGS = None

  # Setting of head
  self.cfg.MODEL.HEADS.NAME = None
  self.cfg.MODEL.HEADS.ARGS = None
        
  # Transforms
  self.cfg.TRANSFORMS = CN()
  self.cfg.TRANSFORMS.TRAIN = None
  self.cfg.TRANSFORMS.TEST = None

  # Optimizer
  self.cfg.OPTIMIZER = CN()
  self.cfg.OPTIMIZER.NAME = "SGD"
  self.cfg.OPTIMIZER.MOMENTUM = 0.9
  self.cfg.OPTIMIZER.WEIGHT_DECAY = 5e-4
  self.cfg.OPTIMIZER.LR = CN()
  self.cfg.OPTIMIZER.LR.backbone = None
  self.cfg.OPTIMIZER.LR.encoding = None
  self.cfg.OPTIMIZER.LR.necks = None
  self.cfg.OPTIMIZER.LR.heads = None
```

The following parameters are about the processing.
```python
  # Train
  self.cfg.ITERATION_NUM = None
  self.cfg.EPOCH_NUM = None
  self.cfg.START_EPOCH = None
  self.cfg.UPDATE_STRATEGY = None
        
  # Validation
  self.cfg.PER_ITERATION = None
  self.cfg.PER_EPOCH = None
  self.cfg.METRICS = None

  # Inference
  self.cfg.FIFTYONE = CN()
  self.cfg.FIFTYONE.NAME = "BirdsTest"
  self.cfg.FIFTYONE.STORE = True

  self.cfg.INTERPRETER = CN()
  self.cfg.INTERPRETER.NAME = "cam"
  self.cfg.INTERPRETER.METHOD = "gradcam"
  self.cfg.INTERPRETER.TARGET_LAYERS = []
```
## The example of the configs

In the ```main.py```, you can import the configs ```from fgvclib.configs import FGVCConfig```, and use it to load config.

```python
import os
import torch 

from fgvclib.configs import FGVCConfig

# load config
    config = FGVCConfig()
    config.load(args.config)
    cfg = config.cfg
    print(cfg)
```