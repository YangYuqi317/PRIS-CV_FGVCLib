# PIM

[PIM](https://arxiv.org/abs/2202.03822)

## Abstract

Visual classification can be divided into coarse-grained and fine-grained classification. Coarse-grained classification represents categories with a large degree of dissimilarity, such as the classification of cats and dogs, while fine-grained classification represents classifications with a large degree of similarity, such as cat species, bird species, and the makes or models of vehicles. Unlike coarse-grained visual classification, fine-grained visual classification often requires professional experts to label data, which makes data more expensive. To meet this challenge, many approaches propose to automatically find the most discriminative regions and use local features to provide more precise features. These approaches only require image-level annotations, thereby reducing the cost of annotation. However, most of these methods require two- or multi-stage architectures and cannot be trained end-to-end. Therefore, we propose a novel plug-in module that can be integrated to many common backbones, including CNN-based or Transformer-based networks to provide strongly discriminative regions. The plugin module can output pixel-level feature maps and fuse filtered features to enhance fine-grained visual classification. Experimental results show that the proposed plugin module outperforms state-of-the-art approaches and significantly improves the accuracy to 92.77\% and 92.83\% on CUB200-2011 and NABirds, respectively.

## Framework

<div align=center>
<img src="https://github.com/YangYuqi317/FGVCLib_docs/blob/main/src/mcl_loss.jpg?raw=true"/>
</div>

Schematic flow of the proposed plug-in module. Backbone Blockk
represents the kth block in the backbone network. When the image is input
to the network, the feature map output by each block will be input into the
Weakly Supervised Selector to screen out areas with strong discrimination or
areas that are less related to classification. Finally, a Combiner is used to fuse
the features of the selected results to obtain the prediction results. The Lfinal
represents the loss function.