# 准备工作

在这个部分，we demonstrate how to prepare an environment with PyTorch.
FGVCLib works on Linux. It requires Python 3.7+,CUDA 10.0+ and PyTorch.

```{note}
If you are experienced with PyTorch and have already installed it, just skip this part and jump to the [next section](#installation). Otherwise, you can follow these steps for the preparation.
```
**Step 0.** Download and install Anaconda from the [official website](https://www.anaconda.com/).

**Step 1.** Create a conda environment and activate it.

```shell
conda create -n fgvclib python=3.7
conda activate fgvclib
```
**Step 2.** Install Pytorch following [official website](https://pytorch.org/), e.g.

On GPU platforms:

```shell
conda install pytorch torchvision -c python
```

# Installation

## Best practices

We recommend that users follow our best practices to install FGVCLib. And FGVCLib needs some requirements to install.

**Step 0.** Install FGVCLib

```shell
git clone https://github.com/dongliangchang/Fine-grained-Visual-Analysis-Library.git
cd Fine-grained-Visual-Analysis-Library.git
```
**Step 1.** Install the requirments

```shell
pip install -r requirements.txt
```
## Trouble shooting

```{note}
Maybe you will meet problems when you install the 'fiftyone', if you have the trouble, you can refer to the following.
```
If the version of Ubuntu >= 18.04, you can execute

```shell
pip install fiftyone
```
If the version of Ubuntu < 18.04, you can execute 

```shell
pip install fiftyone-db-ubuntu1604
```
If you have the error "error while loading shared libraries: libcurl.so.4: cannot open shared object file: No such file", please check whether there is `curl` or not. If you don't have the `curl`, please execute

```shell
sudo apt-get update
sudo apt-get install libcurl4-openssl-dev
sudo apt-get install curl
```