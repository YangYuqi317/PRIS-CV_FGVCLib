# Tutorial 4: Learn about datasets

在fgvclib中，我们主要使用鸟类的数据集：`CUB_200_2011`
我们建立这个文件夹去加载数据集，我们定义了`get_dataset`函数，通过给定的数据集名称，返回对应的数据集。

```python
def get_dataset(dataset_name) -> FGVCDataset:
    r"""Return the dataset with the given name.

        Args: 
            dataset_name (str): 
                The name of dataset.
        
        Return: 
            The dataset contructor method.
    """

    if dataset_name not in globals():
        raise NotImplementedError(f"Dataset {dataset_name} not found!\nAvailable datasets: {available_datasets()}")
    return globals()[dataset_name]
```

## FGVC数据集

首先，我们应该知道我们具有什么数据集。我们定义了函数`available_datasets`来展示所有的可用的FGVC数据集，这个函数将会返回所有可用的FGVC数据集列表。

然后，我们建立了一个类`FGVCDataset`作为`CUB_200_2011`类的输入，`CUB_200_2011`类被用来加载CUB_200_2011数据集。

我们列出了和对应数据集相关的下载链接，关于CUB_200_2011数据集的分支文件夹、文件。

如果你没有相应的数据集，请讲参数`download`设为 **true**
```python
    name: str = "Caltech-UCSD Birds-200-2011"
    link: str = "http://www.vision.caltech.edu/datasets/cub_200_2011/"
    download_link: str = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
    category_file: str = "CUB_200_2011/CUB_200_2011/classes.txt"
    annotation_file: str = "CUB_200_2011/CUB_200_2011/image_class_labels.txt"
    image_dir: str = "CUB_200_2011/CUB_200_2011/images/"
    split_file: str = "CUB_200_2011/CUB_200_2011/train_test_split.txt"
    images_list_file: str = "CUB_200_2011/CUB_200_2011/images.txt" 
```


