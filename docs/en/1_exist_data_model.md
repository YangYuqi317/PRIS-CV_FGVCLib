# 1: Train with the existing models and satndard datastes

To evaluate a model’s accuracy, one usually tests the model on some standard datasets. FGVCLib supports the public datasets including CUB_200_2011. This section will show how to test existing models on supported datasets.

The basic steps are as below:
    1.Prepare the dataset
    2.Prepare a config
    3.Train, test models on the dataset

## The existing models

We provide a variety of existing methods, they are `baseline_resnet50`, `MCL`, `PMG`, `PMG_v2`, `API-Net`, `CAL`, `PIM`, `TransFG`.

In the future we will continue to reproduce new methods and add them into FGVCLib.

## Prepare the dataset
We provide the CUB-200-2011, and we split the dataset into train and test folder.

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

If you have prepared the dataset, you can skip the step1.

**step1**: open the "/fgvclib/datasets/cub.py", and modify the `class CUB_200_2011: __init__ : download:bool=False` to `class CUB_200_2011: __init__ : download:bool=True`
```note
The parameter 'download' controls whether the dataset is downloaded. Directly downloading CUB dataset by setting download=True. Default is False.
```

**step2**: open the "/configs/xxx/xxx.yml", and replace the `DATASET-ROOT` with your own path.

## Train

**step1**: open the "/configs/xxx/xxx.yml", and replace the `WEIGHT-SAVE_DIR` with your own path.
**step2**: open the "/configs/xxx/xxx.yml", and check the configs about the model. You can change the configs by yourself.
**stpe3**: execute main program to train.

```python
python main.py --config configs/resnet/resnet50.yml
```
There are several arguments to control the program.
- '--config': the path of configuration file.
- '--task': train or predict. The default is **train**.
- '--device': two choices are cuda and cpu. The default is **cuda**.
- '--world-size': the number of distributed processes. The default is 4.
- '--dist-url': url used to set up distributed training. The default is 'env://'.

If you want to run it on cpu, you should execute the following：
```python
python main.py --config configs/resnet/resnet50.yml --device cpu
```

## Test
```python
python main.py --config configs/resnet/resnet50.yml --task predict
```