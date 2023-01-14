# Learn about utils
We add some tools for the fgvc, and the tools are about interpreter, logger, learning rate schedules, updating strategy, and visualization.

## Interpreter

We chose the class activation map tool.
We design a class named CAM, the class actication map tool is for explaning the classification result.
All methods are from (pytorch_grad_cam)[git@github.com:jacobgil/pytorch-grad-cam.git]. The methods are gradcam, hirescam, scorecam, gradcam++, xgradcam, eigencam, eigengrafcam, layercam, fullgrad, gradcamelementeise.

There are some args for the class CAM:

- `model (nn.Module)`: The FGVC model
- `target_layers (list)`: The layers used to get CAM weights
- `use_cuda (bool)`: Wheter use gpu
- `method (str)`: The available CAM methods
- `aug_smooth (str)`: The smooth method has the effect of better centering the CAM around the objects
- `eigen_smooth (str)`: The smooth method has the effect of removing a lot of noise.

In "fgvclib/utils/interpreter/__init__.py", we define a function named get_interpreter to return the interpreter with the given name. And the given name is `cam`.

```python
def get_interpreter(interpreter_name):
    r"""
        Args: 
            interpreter_name (str): 
                The name of interpreter.
        
        Return: 
            The interpreter contructor method.
    """
    if interpreter_name not in globals():
        raise NotImplementedError(f"Interpreter not found: {interpreter_name}\nAvailable interpreters: {__all__}")
    return globals()[interpreter_name]
```

### The example
It is used to build interpreter.

```python
gvclib.utils.interpreter import get_interpreter, Interpreter

def build_interpreter(model: nn.Module, cfg: CfgNode) -> Interpreter:
    r"""
    Args:
        cfg (CfgNode): The root config node.
    Returns:
        Interpreter: A Interpreter.
    """
    return get_interpreter(cfg.INTERPRETER.NAME)(model, cfg)
```

## Logger

We define two types logger, `txt logger` and `wandb logger`.

In "fgvclib/utils/logger/__init__.py" we define a function named `get_logger` to return the logger with the given name, and the given names are `wandb_logger`, `txt_logger`

```python
def get_logger(logger_name):
    r"""Return the logger with the given name.

        Args: 
            logger_name (str): 
                The name of logger.
        
        Return: 
            The logger contructor method.
    """

    if logger_name not in globals():
        raise NotImplementedError(f"Logger not found: {logger_name}\nAvailable loggers: {__all__}")
    return globals()[logger_name]
```

### The example 
It can be used to build a logger object or generate the logger.

```python
def build_logger(cfg: CfgNode) -> Logger:
    r"""Build a Logger object according to config.

    Args:
        cfg (CfgNode): The root config node.
    Returns:
        Logger: The Logger object.
    """

    return get_logger(cfg.LOGGER.NAME)(cfg)
```

## Learning rate schedules

In "fgvclib/utils/lr_schedules/__init__.py" we define a function named `get_lr_schedule` to return the learning rate schedule with the given name, and the given name is `cosine_anneal_schedule`.

```python
def get_lr_schedule(lr_schedule_name):
    r"""Return the learning rate schedule with the given name.

        Args: 
            lr_schedule_name (str): 
                The name of learning rate schedule.
        
        Return: 
            The learning rate schedule contructor method.
    """

    if lr_schedule_name not in globals():
        raise NotImplementedError(f"Learning rate schedule not found: {lr_schedule_name}\nAvailable learning rate schedules: {__all__}")
    return globals()[lr_schedule_name]
```

And we define the function named `cosine_anneal_schedule`

```python
def cosine_anneal_schedule(optimizer, current_epoch, total_epoch):
    cos_inner = np.pi * (current_epoch % (total_epoch)) 
    cos_inner /= (total_epoch)
    cos_out = np.cos(cos_inner) + 1
    
    for i in range(len(optimizer.param_groups)):
        current_lr = optimizer.param_groups[i]['lr']
        optimizer.param_groups[i]['lr'] = float(current_lr / 2 * cos_out)
```

### The example
It can be used in the file `main.py` for the processing of training.

```python
from fgvclib.utils.lr_schedules import cosine_anneal_schedule

   cosine_anneal_schedule(optimizer, epoch, cfg.EPOCH_NUM)

```

## Update strategy
We provide three types update strategy contructor methods, `progressive updating with jigsaw`, `progressive updating consistency constraint`, and `general updating`.

**progressive updating with jigsaw**: For more details about progressive updating with jigsaw, see "fgvclib/utils/update_strategy/progressibe_updating_with_jigsaw.py".

**progressive updating consistency constraint**: For more details about progressive updating consistency constraint, see "fgvclib/utils/update_strategy/progressive_updating_consistency_constraint.py".

**general updating**: For more details about general updating, see "fgvclib/utils/update_strategy/general_updating.py".

In "fgvclib/utils/update_strategy/__init__.py", we define a function named `get_update_strategy` to return the update stratrgy contructor method with the given name. And the given names are `progressive_updating_with_jigsaw`, `progressive_updating_consistency_constraint`, `general_updating`

```python
def get_update_strategy(strategy_name):
    r"""
        Args: 
            strategy_name (str): 
                The name of the update strategy.
        
        Return: 
            The update strategy contructor method.
    """

    if strategy_name not in globals():
        raise NotImplementedError(f"Strategy not found: {strategy_name}\nAvailable strategy: {__all__}")
    return globals()[strategy_name]

```

### The example 

The update stratrgy contructor method is used to update the FGCV model, so we can import it when update model.

In "fgvclib/apis/update_model.py", we import fgvclib.utils.update_strategy.

```python
from fgvclib.utils.update_strategy import get_update_strategy
from fgvclib.utils.logger import Logger

def update_model(model: nn.Module, optimizer: Optimizer, pbar:Iterable, strategy:str="general_updating", use_cuda:bool=True, logger:Logger=None):
    model.train()
    mean_loss = 0.
    for batch_idx, train_data in enumerate(pbar):
        losses_info = get_update_strategy(strategy)(model, train_data, optimizer, use_cuda)
        mean_loss = (mean_loss * batch_idx + losses_info['iter_loss']) / (batch_idx + 1)
        losses_info.update({"mean_loss": mean_loss})
        logger(losses_info, step=batch_idx)
        pbar.set_postfix(losses_info)
```

## Visualization

We designed this module to visualize the results. This module can help to show the heat map, which is better for the result. In this module, `fiftyone` is mainly imported and we create a class named `VOXEL`.
```python
class VOXEL:

    def __init__(self, dataset, name:str, persistent:bool=False, cuda:bool=True, interpreter:Interpreter=None) -> None:
        self.dataset = dataset
        self.name = name
        self.persistent = persistent
        self.cuda = cuda
        self.interpreter = interpreter

        if self.name not in self.loaded_datasets():
            self.fo_dataset = self.create_dataset()
            self.load()
        else:
            self.fo_dataset = fo.load_dataset(self.name)

        self.view = self.fo_dataset.view() 

    def create_dataset(self) -> fo.Dataset:
        return fo.Dataset(self.name)

    def loaded_datasets(self) -> t.List:
        return fo.list_datasets()

    def load(self):
        
        samples = []

        for i in tqdm(range(len(self.dataset))):
            path, anno = self.dataset.get_imgpath_anno_pair(i)

            sample = fo.Sample(filepath=path)

            # Store classification in a field name of your choice
            sample["ground_truth"] = fo.Classification(label=anno)

            samples.append(sample)

            # Create dataset
        
        self.fo_dataset.add_samples(samples)
        self.fo_dataset.persistent = self.persistent

    def predict(self, model:nn.Module, transforms, n:int=inf, name="prediction", seed=51, explain:bool=False):
        model.eval()
        if n < inf:
            self.view = self.fo_dataset.take(n, seed=seed)

        with fo.ProgressBar() as pb:
            for sample in pb(self.view):
                image = Image.open(sample.filepath)
                image = transforms(image).unsqueeze(0)
                
                if self.cuda:
                    image = image.cuda()
                    pred = model(image)
                    index = torch.argmax(pred).item()
                    confidence = pred[:, index].item()

    
                sample[name] = fo.Classification(
                    label=str(index),
                    confidence=confidence
                )

                if self.interpreter:
                    heatmap = self.interpreter(image_path=sample.filepath, image_tensor=image, transforms=transforms)
                    sample["heatmap"] = fo.Heatmap(map=heatmap)

                sample.save()
        print("Finished adding predictions")
```