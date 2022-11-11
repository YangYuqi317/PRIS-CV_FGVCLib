## Introduction

FGVCLib is an open-source and well documented library for Fine-grained Visual Analysis. It is based on Pytorch with performance and friendly API. Our code is pythonic, and the design is consistent with torchvision. You can easily develop new algorithms, or readily apply existing algorithms.
The branch works with **torch 1.12.1**, **torchvision 0.13.1**.

<details open>
<summary>Major features</summary>

- **Modular Design**

  We decompose the detection framework into different components and one can easily construct a customized object detection framework by combining different modules.

- **State of the art**
  We implement state-of-the-art methods by the FGVCLib, [PMG](https://arxiv.org/abs/2003.03836v3), [MCL](https://arxiv.org/abs/2002.04264). 


## Installation

Please refer to [Installation](docs/get_started.md/#installation) for installation instructions.

## Getting started 

Please see [get_started.md](docs/get_started.md) for the basic usage of FGVCLib. We provide the tutorials for:

- [with existing data existing model](docs/1_exist_data_model.md)
- [with existing data new model](docs/2_exist_data_new_model.md)
- [learn about apis](docs/tutorials/tutorial1_apis.md)
- [learn about configs](docs/tutorials/tutorial2_configs.md)
- [learn about criterions](docs/tutorials/tutorial3_criterions.md)
- [learn about datasets](docs/tutorials/tutorial4_datasets.md)
- [learn about metrics](docs/tutorials/tutorial5_metrics.md)
- [learn about model](docs/tutorials/tutorial6_model.md)
- [learn about transforms](docs/tutorials/tutorial7_transform.md)
- [learn about the tools](docs/useful_tools.md)


</details>

## Overview of Benchmark and Model Zoo

<div align="center">
  <b>Architectures</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Fine-grained Visual Classification</b>
      </td>
      <td>
        <b>Other</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
            <li><a href="configs/baseline_resnet50">Baseline_ResNet50</a></li>
            <li><a href="configs/mcl_vgg16">MCL_VGG16</a></li>
            <li><a href="configs/pmg_resnet50">PMG_ResNet50</a></li>
            <li><a href="configs/pmg_v2_resnet50">PMG_V2_ResNet50</a></li>
      </ul>  
      </td>
      <td>
        <ul>
            <li>visualization</li>
      </ul>  
      </td>
    </tr>
  </tbody>
</table>

<div align="center">
  <b>Components</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Backbones</b>
      </td>
      <td>
        <b>Encoders</b>
      </td>
      <td>
        <b>Heads</b>
      </td>
      <td>
        <b>Necks</b>
      </td>
      <td>
        <b>Sotas</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
            <li>Resnet</li>
            <li>VGG</li>
      </ul>  
      </td>
      <td>
        <ul>
            <li>Global Max Pooling</li>
            <li>Global Avg Pooling</li>
            <li>Max Pooling 2d</li>
      </ul>  
      </td>
      <td>
        <ul>
            <li>Classifier_1_FC</li>
            <li>Classifier_2_FC</li>
      </ul>  
      </td>
      <td>
        <ul>
            <li>Multi-scale Convolution neck</li>
      </ul>  
      </td>
      <td>
        <ul>
            <li>Baseline</li>
            <li><a href="configs/mcl_vgg16/README.md">MCL</a></li>
            <li><a href="configs/pmg_resnet50/README.md">PMG</li>
            <li>PMG_v2</li>
      </ul>  
      </td>
    </tr>
  </tbody>
</table>