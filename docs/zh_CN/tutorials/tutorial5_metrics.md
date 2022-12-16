# 教程 5: 学习评价指标文件

我们提供了三种评价标准：准确率`accuracy`、精确率`precision`、召回率`recall`作为训练和测试的结果。从"Torchmetrics"中调用者三种评价指标，同时，在"__init__"中设置了评价指标的列表`__all__ = ["accuracy", "precision", "recall"]`

关于准确率**accuracy**参数的更多细节参见[torchmetrics.Accuracy object](https://torchmetrics.readthedocs.io/en/stable/classification/accuracy.html)

关于精确率**precision**参数的更多细节参见[torchmetrics.Precision object](https://torchmetrics.readthedocs.io/en/stable/classification/precision.html)

关于召回率**recall**参数的更多细节参见[torchmetrics.Recall object](https://torchmetrics.readthedocs.io/en/stable/classification/recall.html)

## 准确率Accuracy

准确率`accuracy`被定义为：`accuracy(name:str="accuracy(top-1)", top_k:int=1, threshold:float=None)`
- 参数:

  `"name(str)"`: 评价指标的名称, 比如 accuracy(top-1)
  `"top_k (int)"`: 找到正确标签时的最高概率或logit分数预测的数量
  `"threshhold (float, optional)"`: 在二进制或多标签输入的情况下，将概率或logit预测转换为二进制（0，1）预测的阈值

- 返回值:

  `NamedMetirc`: 自定义名称的torchmetrics度量


## 精确率Precision

精确率`precision`被定义为`precision(name:str="precision(threshold=0.5)", top_k:int=None, threshold:float=0.5)`

- 参数:

  `"name(str)"`: 评价指标的名称, 比如 accuracy(top-1)
  `"top_k (int)"`: 找到正确标签时的最高概率或logit分数预测的数量
  `"threshhold (float, optional)"`: 在二进制或多标签输入的情况下，将概率或logit预测转换为二进制（0，1）预测的阈值

- 返回值:

  `NamedMetirc`: 自定义名称的torchmetrics度量

## 召回率Recall

召回率`recall`被定义为`recall(name:str="recall(threshold=0.5)", top_k:int=None, threshold:float=0.5)`

- 参数:

  `"name(str)"`: 评价指标的名称, 比如 accuracy(top-1)
  `"top_k (int)"`: 找到正确标签时的最高概率或logit分数预测的数量
  `"threshhold (float, optional)"`: 在二进制或多标签输入的情况下，将概率或logit预测转换为二进制（0，1）预测的阈值

- 返回值:

  `NamedMetirc`: 自定义名称的torchmetrics度量

## 举例

### 为评估构建度量标准
```python
from fgvclib.metrics import get_metric
from fgvclib.metrics import NamedMetric

def build_metrics(metrics_cfg: CfgNode, use_cuda:bool=True) -> t.List[NamedMetric]:

    metrics = []
    for cfg in metrics_cfg:
        metric = get_metric(cfg["metric"])(name=cfg["name"], top_k=cfg["top_k"], threshold=cfg["threshold"])
        if use_cuda:
            metric = metric.cuda()
        metrics.append(metric)
    return metrics
```

### 评估FGVC模型
```python
def evaluate_model(model:nn.Module, p_bar:t.Iterable, metrics:t.List[NamedMetric], use_cuda:bool=True) -> t.Dict:

    model.eval()
    results = dict()
    
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(p_bar):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            for metric in metrics:
                _ = metric.update(model(inputs), targets) 
    
    for metric in metrics:
        result = metric.compute()
        results.update({
            metric.name: round(result.item(), 3)
        })

    return results
```

### 准确率的输出
In the processing of train:
```python
acc = evaluate_model(model, test_bar, metrics=cfg.METRICS, use_cuda=cfg.USE_CUDA)
logger("Evalution Result:")
logger(acc)
```

In the processing of predict:
```python
metrics = build_metrics(cfg.METRICS)
acc = evaluate_model(model, pbar, metrics=metrics, use_cuda=cfg.USE_CUDA)

print(acc)
```
