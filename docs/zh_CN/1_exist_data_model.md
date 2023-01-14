# 1：在标准数据集上训练已有模型

为了评估模型的准确性，人们通常在一些标准数据集上测试模型。FGVCLib支持包括CUB_200_2011在内的公共数据集。本节将展示如何在受支持的数据集上测试现有模型。

基本步骤如下：
    1.准备标准数据集
    2.准备配置文件
    3.在标准数据集上对模型进行训练、测试和预测

## 已有模型
我们提供了多种已有的方法，它们分别是：`baseline_resnet50`, `MCL`, `PMG`, `PMG_v2`, `API-Net`, `CAL`, `PIM`, `TransFG`。

今后我们将会继续复现更多新的方法并将它们更新至FGVCLib中。

## 准备标准数据集
我们提供了CUB-200-2011，我们将数据集分为训练文件夹和测试文件夹。

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

如果你已经准备好数据集了，你可以跳过下面的第一步。

**第一步**：打开"/fgvclib/datasets/cub.py"，将`class CUB_200_2011: __init__ : download:bool=False`修改为`class CUB_200_2011: __init__ : download:bool=True`
```note
参数“download”控制是否下载数据集。通过设置download=True直接下载CUB数据集。默认为False。
```

**第二步**：打开"/configs/xxx/xxx.yml"，将`DATASET-ROOT`替换为你自己的路径。

## 训练模型
**第一步**：打开"/configs/xxx/xxx.yml"，将`WEIGHT-SAVE_DIR`替换为你自己的路径。
**第二步**：打开"/configs/xxx/xxx.yml"，检查模型的配置，你可以自己修改这些配置。
**第三步**：执行主程序main.py进行训练

```python
python main.py --config configs/resnet/resnet50.yml
```

这里存在几类参数控制着程序的运行配置：
- '--config':配置文件路径。
- '--task': 默认为**train**。
- '--device'：两种选择是cuda和cpu。默认为**cuda**。
- '--world-size'：分布式进程的数量。默认值是4。
- '--dist-url':Url用于设置分布式培训。默认值是'env://'。

如果你想在cpu上运行它，你应该执行下面的：
```python
python main.py --config configs/resnet/resnet50.yml --device cpu
```

## 测试模型
```python
python main.py --config configs/resnet/resnet50.yml --task predict
```