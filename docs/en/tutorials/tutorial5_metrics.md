# Tutorial 5: Learn about metrics

We provide 3 metrics, `accuracy`, `precision`, `recall` as the results of training and testing. They are from the Torchmetrics, and in "__init__" we set the list of them `__all__ = ["accuracy", "precision", "recall"]`

For details about the meanings of the **accuracy** parameters, see [torchmetrics.Accuracy object](https://torchmetrics.readthedocs.io/en/stable/classification/accuracy.html).

For details about the meanings of the **precision** parameters, see [torchmetrics.Precision object](https://torchmetrics.readthedocs.io/en/stable/classification/precision.html).

For details about the meanings of the **recall** parameters, see [torchmetrics.Recall object](https://torchmetrics.readthedocs.io/en/stable/classification/recall.html).


## Accuracy
The `accuracy` is defined as `accuracy(name:str="accuracy(top-1)", top_k:int=1, threshold:float=None)`
- Args:

  `"name(str)"`: The name of metric, e.g. accuracy(top-1)
  `"top_k (int)"`: Number of the highest probability or logit score predictions considered finding the correct label.
  `"threshhold (float, optional)"`: Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case of binary or multi-label inputs.

- Return:

  `NamedMetirc`: A torchmetrics metric with customed name.


## Precision
The `precision` is defined as `precision(name:str="precision(threshold=0.5)", top_k:int=None, threshold:float=0.5)`
- Args:

  `"name(str)"`: The name of metric, e.g. accuracy(top-1)
  `"top_k (int)"`: Number of the highest probability or logit score predictions considered finding the correct label.
  `"threshhold (float, optional)"`: Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case of binary or multi-label inputs.

- Return:

  `NamedMetirc`: A torchmetrics metric with customed name.

## Recall
The `recall` is defined as `recall(name:str="recall(threshold=0.5)", top_k:int=None, threshold:float=0.5)`
- Args:

  `"name(str)"`: The name of metric, e.g. accuracy(top-1)
  `"top_k (int)"`: Number of the highest probability or logit score predictions considered finding the correct label.
  `"threshhold (float, optional)"`: Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case of binary or multi-label inputs.

- Return:

  `NamedMetirc`: A torchmetrics metric with customed name.

## The example

### Build metrics for evaluation.
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

### Evaluate the FGVC model.
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

### Output the accuracy.
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
