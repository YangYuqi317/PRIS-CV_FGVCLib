# The conigs of the state-of-the-art methods

In "/configs/", we write four file to save the configs of the state-of-the-art methods. And the file type is `.yml`
The configs combine `experiment name`, `weight`, `logger`, `dataset`, `model`, `transforms`, `optimizer`, and Validation details.

Take the **"baseline_resnet50"** for example.

- Experiment name 

   **EXP_NAME**: the name of this method.
   ```python
   EXP_NAME: "Baseline_ResNet50"
   ```

- Resume weight

   **RESUME_WEIGHT** is set ~
   ```python
   RESUME_WEIGHT: ~
   ```

- Logger
  
   **LOGGER**: the logger of this process.
   ```python
   LOGGER: 
     NAME: "txt_logger"
   ```

- Dataset

   **DATASET**: the dataset
   - NAME: the name of the dataset
   - ROOT: the root of the dataset, **you should change it to your own root in advance**
   - TRAIN: the parameters about the processing of training
     - BATCH_SIZE: the batch size
     - POSITIVE: the positive
     - PIN_MEMORY: is bool, True or False
     - SHUFFLE: is bool, True or False
     - NUM_WORKERS: the number of workers
   - TEST: the parameters about the processing of testing
     - BATCH_SIZE: the batch size
     - POSITIVE: the positive
     - PIN_MEMORY: is bool, True or False
     - SHUFFLE: is bool, True or False
     - NUM_WORKERS: the number of workers

```python
DATASET:
  NAME: "CUB_200_2011"
  ROOT: "/data/wangxinran/dataset/"
  TRAIN: 
    BATCH_SIZE: 2
    POSITIVE: 0
    PIN_MEMORY: True
    SHUFFLE: True
    NUM_WORKERS: 4
  TEST: 
    BATCH_SIZE: 2
    POSITIVE: 0
    PIN_MEMORY: False
    SHUFFLE: False
    NUM_WORKERS: 4
```

- Model

   **MODEL**: the model
   - NAME: the name of the model
   - CLASS_NUM: the number of the class
   - CRITERIONS: the loss
     - name: the name of loss function
     - args: the args are determined by loss 
     - w: the weight
   - BACKBONE: the backbone of the model, the parameters about the args are set according the corresonding file.
     - NAME: the name of the bockbone
     - ARGS: 
       - pretrained: is bool, True or False
       - del_keys: is a list of the del_keys
   - ENCODING: the encoding of the model, the parameters about the args are set according the corresonding file.
     - NAME: the name of the encoding
   - NECKS: the neck of the model
     - NAME: the name of the neck
   - HEADS: the head of the model, the parameters about the ARGS are set according the corresonding file.
     - NAME: the name of the head
     - ARGS: the args are determined by the classifier.

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
  ENCODING:
    NAME: "global_avg_pooling"
  NECKS:
    NAME: ~
  HEADS:
    NAME: "classifier_1fc"
    ARGS:
      - in_dim: 
        - 2048
```
- Transforms

   **TRANSFORMS**: the transforms
   - TRAIN: the parameters about the process of training
     - name: the name of the transform type
     - other parameters: according the transform type to set the corresonding parameters.
   - TEST: the parameters about the process of testing
     - name: the name of the transform type
     - other parameters: according the transform type to set the corresonding parameters.

```python
TRANSFORMS: 
  TRAIN: 
    - name: "resize"
      size: 
        - 600
        - 600
    - name: "random_crop"
      size: 448
      padding: 8
    - name: "random_horizontal_flip"
      prob: 0.5
    - name: "to_tensor"
    - name: "normalize"
      mean: 
         - 0.5
         - 0.5
         - 0.5
      std: 
        - 0.5
        - 0.5
        - 0.5
  TEST:
    - name: "resize"
      size: 
        - 600
        - 600
    - name: "center_crop"
      size: 448
    - name: "to_tensor"
    - name: "normalize"
      mean: 
         - 0.5
         - 0.5
         - 0.5
      std: 
        - 0.5
        - 0.5
        - 0.5
```

- Optimizer

   **OPTIMIZER**: the optimizer 
   - NAME: the name of the optimizer
   - MOMENTUM: the momentum of the optimizer
   - LR: the learning rate 
     - backbone: the learning rate of the backbone
     - encoding: the learning rate of the encoding
     - necks: the learning rate of the neck
     - heads: the learning rate of the head

```python
OPTIMIZER:
  NAME: "SGD"
  MOMENTUM: 0.9
  LR: 
    backbone: 0.0002
    encoding: 0.002
    necks: 0.002
    heads: 0.002
```

- Iteration number

   **ITERATION_NUM**: the Iteration number

```python
ITERATION_NUM: ~
```

- Epoch number

   **EPOCH_NUM**: the epcoh number

```python
START_EPOCH: 0
```

- Update strategy

   **UPDATE_STRATEGY**: the update strategy

```python
UPDATE_STRATEGY: "general_updating"
```

- Per iteration

   **PER_ITERATION**: the per iteration

```python
PER_ITERATION: ~
```

- Per epoch

   **PER_EPOCH**： the per per iteration

```python
PER_EPOCH: ~
```

- Metrics

   **METRICS**: the metrics
   - name: the name of the metric type
   - other parameters: according the metrics type to set the corresonding parameters.

```python
METRICS: 
  - name: "accuracy(topk=1)"
    metric: "accuracy"
    top_k: 1
    threshold: ~
  - name: "accuracy(topk=5)"
    metric: "accuracy"
    top_k: 5
    threshold: ~
  - name: "recall(threshold=0.5)"
    metric: "recall"
    top_k: ~
    threshold: 0.5
  - name: "precision(threshold=0.5)"
    metric: "precision"
    top_k: ~
    threshold: 0.5
```

- Interpreter

   **INTERPRETER**: the interpreter
   - NAME: the name of the interpreter
   - METHOD: the method of the interpreter
   - TARGET_LAYERS: the target layers