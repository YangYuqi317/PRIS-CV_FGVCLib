# Tutorial 4: Learn about datasets
In FGVCLib, we mainly use the Birds dataset, `CUB_200_2011`.
We build this folder to load the dataset. We define the function `get_dataset` to get the dataset dict, and using the function `dataset` to return the dataset

```python
def get_dataset(dataset_name) -> FGVCDataset:
    r"""Return the dataset with the given name.

        Args: 
            dataset_name (str): 
                The name of dataset.
        
        Return: 
            The dataset contructor method.
    """
```

```python
def dataset(name):
    
    def register_function_fn(cls):
        if name in __DATASET_DICT__:
            raise ValueError("Name %s already registered!" % name)
        if not issubclass(cls, FGVCDataset):
            raise ValueError("Class %s is not a subclass of %s" % (cls, FGVCDataset))
        __DATASET_DICT__[name] = cls
        return cls

    return register_function_fn

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module_name = file[:file.find('.py')]
        module = importlib.import_module('fgvclib.datasets.' + module_name) 
```

## FGVC Dataset
Firstly, we should know what dataset we have. We define the function `available_datasets` to show all available FGVC datasets, and this function will return a list wih all available FGVC datasets.

Then, we set a class `FGVCDataset` as the input of class `CUB_200_2011` which is used for loading the CUB_200_2011 dataset.

CUB_200_2011 is the Caltech-UCSD Birds-200-2011 dataset.
We list the relevant link, file, and dir about CUB_200_2011.

If you don't have the dataset, please set the `download` **true**.

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

- Args:
  
  root (str): The root directory of CUB dataset.
  mode (str): The split of CUB dataset.
  download (bool): Directly downloading CUB dataset by setting download=True. Default is False.
  transforms (torchvision.transforms.Compose): The PyTorch transforms Compose class used to preprocessing the data.
  positive (int): If positive = n > 0, the __getitem__ method will an extra list of n images of same category.

