# Learn about utils

我们为FGVCLib添加了一些工具，这些工具包括解释器、记录器、学习率表、更新策略和可视化。

## 解释器

我们选择了类激活映射工具，我们设计了一个名为CAM的类，累计或映射工具用于解释分类结果。
所有的方法君来自于(pytorch_grad_cam)[git@github.com:jacobgil/pytorch-grad-cam.git]。方法有：gradcam, hirescam, scorecam, gradcam++, xgradcam, eigencam, eigengrafcam, layercam, fullgrad, gradcamelementeise.

以下是关于类CAM的一些参数：

- `model (nn.Module)`: FGVC模型
- `target_layers (list)`: 该层用于得到CAM权重
- `use_cuda (bool)`: 是否使用gpu
- `method (str)`: 可用的CAM方法
- `aug_smooth (str)`: 平滑法具有更好的使CAM围绕物体居中的作用
- `eigen_smooth (str)`: 平滑法具有移动噪声的作用

在"fgvclib/utils/interpreter/__init__.py"中，我们定义了函数`get_interpreter`，根据给定的名称返回对应的解释器，给定的名称有：`cam`

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

### 举例

以构建一个解释器为例

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

## 记录器

我们定义了两种记录器，`txt logger`和`wandb logger`

在"fgvclib/utils/logger/__init__.py"中，我们定义了一个函数`get_logger`，根据给定的名称返回对应的记录器，个icing的名称有：`wandb_logger`, `txt_logger`

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

### 举例

它可以用于构建记录器对象或生成记录器

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

## 学习率表

在"fgvclib/utils/lr_schedules/__init__.py"中，我们定义了一个函数`get_lr_schedule`，根据给定的名称返回对应的学习率表，给定的名称有：`cosine_anneal_schedule`

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

并且，我们定义了函数 `cosine_anneal_schedule`

```python
def cosine_anneal_schedule(optimizer, current_epoch, total_epoch):
    cos_inner = np.pi * (current_epoch % (total_epoch)) 
    cos_inner /= (total_epoch)
    cos_out = np.cos(cos_inner) + 1
    
    for i in range(len(optimizer.param_groups)):
        current_lr = optimizer.param_groups[i]['lr']
        optimizer.param_groups[i]['lr'] = float(current_lr / 2 * cos_out)
```

### 举例

可以在`main.py`文件中，在训练过程中使用它

```python
from fgvclib.utils.lr_schedules import cosine_anneal_schedule

   cosine_anneal_schedule(optimizer, epoch, cfg.EPOCH_NUM)

```

## 更新策略

我们提供了三种类型的更新策略构造方法：`progressive updating with jigsaw`, `progressive updating consistency constraint`, 和 `general updating`

**progressive updating with jigsaw**: 有关用jigsaw渐进式更新的更多详细信息，参见文件"fgvclib/utils/update_strategy/progressibe_updating_with_jigsaw.py"

**progressive updating consistency constraint**: 有关渐进式更新一致性约束的详细信息，参见文件"fgvclib/utils/update_strategy/progressive_updating_consistency_constraint.py"

**general updating**: 有关一般更新的详细信息，参见"fgvclib/utils/update_strategy/general_updating.py"

在"fgvclib/utils/update_strategy/__init__.py"中，我们定义了一个函数`get_update_strategy`，根据给定的名称返回对应的更新策略方法，给出的名称有：`progressive_updating_with_jigsaw`, `progressive_updating_consistency_constraint`, `general_updating`

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

### 举例
在更新模型时导入该模块，使用更新策略构造方法更新FGVC模型

在"fgvclib/apis/update_model.py"中，我们导入了fgvclib.utils.update_strategy

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

## 可视化

我们设计该模块将结果进行可视化,这个模块可以帮助显示热图，帮助我们更好的理解实验结果。在这个模块中，我们导入了'fiftyone'，并且我们创建了一个名为'VOXEL'的类。
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