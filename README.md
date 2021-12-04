<h1 align="center">Pruning Self-attentions into Convolutional Layers in Single Path</h1>

**This is the official repository for our paper:** [Pruning Self-attentions into Convolutional Layers in Single Path](https://arxiv.org/abs/2111.11802) by [Haoyu He](https://charles-haoyuhe.github.io/), [Jing liu](https://sites.google.com/view/jing-liu/%E9%A6%96%E9%A1%B5), [Zizheng Pan](https://zizhengpan.github.io/), [Jianfei Cai](https://jianfei-cai.github.io/), [Jing Zhang](https://scholar.google.com/citations?user=9jH5v74AAAAJ&hl=en), [Dacheng Tao](https://www.sydney.edu.au/engineering/about/our-people/academic-staff/dacheng-tao.html) and [Bohan Zhuang](https://bohanzhuang.github.io/).

***

### Introduction:

To reduce the massive computational resource consumption for ViTs and add convolutional inductive bias, **our SPViT prunes pre-trained ViT models into accurate and compact hybrid models by pruning self-attentions into convolutional layers**. Thanks to the proposed weight-sharing scheme between self-attention and convolutional layers that cast the search problem as finding which subset of parameters to use, our **SPViT has significantly reduced search cost**.

***

### Experimental results:

We provide experimental results and pre-trained models for SPViT. Note that the performance for SPViT-Swin is higher than the ones reported in [here](https://arxiv.org/abs/2111.11802), since we fixed one bug that didn't properly initialize batch-normalization layers.

| Name          | Acc@1 | Acc@5 | # parameters | FLOPs | Model |
| :------------ | :---: | :---: | ------------ | ----- | ----- |
| SPViT-DeiT-Ti | 70.7  | 90.3  | 4.9M         | 1.0G  |       |
| SPViT-DeiT-S  | 78.3  | 94.3  | 16.4M        | 3.3G  |       |
| SPViT-DeiT-B  | 81.5  | 95.6  | 62.3M        | 11.7  |       |

| Name          | Acc@1 | Acc@5 | # parameters | FLOPs | Model |
| :------------ | :---: | :---: | ------------ | ----- | ----- |
| SPViT-Swin-Ti | 80.1  | 94.9  | 26.3M        | 3.3   |       |
| SPViT-Swin-S  | 82.4  | 96.0  | 39.2M        | 6.1   |       |
| SPViT-Swin-B  | 83.2  | 96.3  | 68.0M        | 11.4  |       |

### Getting started:

In this repository, we provide code for pruning two representative ViT models.

- SPViT-DeiT that prunes [DeiT](https://github.com/facebookresearch/deit). Please see [SPViT_DeiT/README.md](SPViT_DeiT/README.md ) for details.
- SPViT-Swin that prunes [Swin](https://github.com/microsoft/Swin-Transformer). Please see [SPViT_Swin/README.md](SPViT_Swin/README.md) for details.

***

If you find our paper useful, please consider cite:

```
@article{he2021Pruning,
  title={Pruning Self-attentions into Convolutional Layersin Single Path},
  author={He, Haoyu and Liu, Jing and Pan, Zizheng and Cai, Jianfei and Zhang, Jing and Tao, Dacheng and Zhuang, Bohan},
  journal={arXiv preprint arXiv:2111.11802},
  year={2021}
}
```

