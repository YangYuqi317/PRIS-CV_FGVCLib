# 准备工作

在这个部分，我们将展示如何搭建Pytorch环境。FGVCLib工作于Linux系统中，并且需要Python 3.7+，CUDA 10.0+, Pytorch。

```{note}
如果你使用过Pytroch并且已经下载好Pytorch，可以越过这一部分，并跳转到[下一个部分](#installation)。如果你没有Pytorch，可以遵循下面的步骤准备环境。
```
**Step 0.** 从[官网](https://www.anaconda.com/)下载并安装Anaconda。

**Step 1.** 创建一个虚拟环境并且激活它。

```shell
conda create -n fgvclib python=3.7
conda activate fgvclib
```
**Step 2.** 从[官网](https://pytorch.org/)上下载并安装Pytorch。

如果你有GPU:

```shell
conda install pytorch torchvision -c python
```

# 安装

## 最佳示例

我们建议开发者遵循我们的最佳示例来安装FGVCLib，FGVCLib需要一些要求和安装包。

**Step 0.** 安装FGVClib

```shell
git clone https://github.com/dongliangchang/Fine-grained-Visual-Analysis-Library.git
cd Fine-grained-Visual-Analysis-Library.git
```
**Step 1.** 安装需要的库

```shell
pip install -r requirements.txt
```
## 问题解答

```{note}
你在安装的过程中可能会遇到一些问题，主要问题是关于安装'fiftyonr'，如果你在安装'fiftyone'时遇到了问题，你可以参考下面的方法。
```
如果你的Ubuntu版本 >=18.04,你可以执行下面的命令

```shell
pip install fiftyone
```
如果你的Ubuntu版本 < 18.04，你可以执行下面的命令 

```shell
pip install fiftyone-db-ubuntu1604
```
如果你遇到了这样的报错"error while loading shared libraries: libcurl.so.4: cannot open shared object file: No such file"，请检查是否有`curl`，如果你没有`curl`,请执行下面的命令

```shell
sudo apt-get update
sudo apt-get install libcurl4-openssl-dev
sudo apt-get install curl
```