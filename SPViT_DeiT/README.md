### Getting started on SPViT-DeiT:

#### Installation and data preparation

- First, you can install the required environments as illustrated in the [DeiT](https://github.com/facebookresearch/deit) repository or follow the instructions below:

  ```bash
  # Create virtual env
  conda create -n spvit-deit python=3.7 -y
  conda activate spvit-deit
  
  # Install PyTorch 1.7.0+ and torchvision 0.8.1+ and pytorch-image-models 0.3.2:
  conda install -c pytorch pytorch torchvision
  pip install timm==0.3.2
  ```

- Next, install some other dependencies that are required by SPViT:

  ```bash
  pip install tensorboardX tensorboard
  ```

- Please refer to the [DeiT](https://github.com/facebookresearch/deit) repository to prepare the standard ImageNet dataset, then link the ImageNet dataset under the `data`folder:

  ```bash
  $ tree data
  imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
  ```

#### Download pretrained models

- We start searching and fine-tuneing both from the pre-trained models.

- Since we provide training scripts for three DeiT models: DeiT-Ti, DeiT-S and DeiT-B, please download the corresponding three pre-trained models from the [DeiT](https://github.com/facebookresearch/deit) repository as well.

- Next, move the downloaded pre-trained models into the following file structure:

  ```bash
  $ tree model
  ├── deit_base_patch16_224-b5f2ef4d.pth
  ├── deit_small_patch16_224-cd65a155.pth
  ├── deit_tiny_patch16_224-a1311bcf.pth
  ```

- Note that do not change the filenames for the pre-trained models as we hard-coded these filenames when tailoring and loading the pre-trained models. Feel free to modify the hard-coded parts when pruning from other pre-trained models.

#### Searching

To search architectures with SPViT-DeiT-Ti, run:

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=3146 --use_env main_pruning.py --config config/spvit_deit_ti_l200_t10_search.json
```

To search architectures with SPViT-DeiT-S, run:

```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port=3146 --use_env main_pruning.py --config config/spvit_deit_sm_l30_t32_search.json
```

To search architectures with SPViT-DeiT-B, run:

```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port=3146 --use_env main_pruning.py --config config/spvit_deit_bs_l006_t100_search.json
```

#### Fine-tuning

You can start fine-tuning from either your own searched architectures or from our provided architectures by modifying and assigning the MSA indicators in `assigned_indicators` and the FFN indicators in `searching_model`.

To fine-tune the architectures searched by SPViT-DeiT-Ti, run:

```bash
python -m torch.distributed.launch --nproc_per_node=4 --master_port=3146 --use_env main_pruning.py --config config/spvit_deit_ti_l200_t10_ft.json
```

To fine-tune the architectures with SPViT-DeiT-S, run:

```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port=3146 --use_env main_pruning.py --config config/spvit_deit_sm_l30_t32_ft.json
```

To fine-tune the architectures with SPViT-DeiT-B, run:

```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port=3146 --use_env main_pruning.py --config config/spvit_deit_bs_l006_t100_ft.json
```

#### Evaluation

We provide several examples for evaluating pre-trained SPViT models.

To evaluate SPViT-DeiT-Ti pre-trained models, run:

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=3146 --use_env main_pruning.py --config config/spvit_deit_ti_l200_t10_ft.json --resume [PRE-TRAINED MODEL PATH] --eval
```

To evaluate SPViT-DeiT-S pre-trained models, run:

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=3146 --use_env main_pruning.py --config config/spvit_deit_sm_l30_t32_ft.json --resume [PRE-TRAINED MODEL PATH]  --eval
```

To evaluate SPViT-DeiT-B pre-trained models, run:

```bash
python -m torch.distributed.launch --nproc_per_node=1 --master_port=3146 --use_env main_pruning.py --config config/spvit_deit_bs_l006_t100_ft.json --resume [PRE-TRAINED MODEL PATH] --eval
```

#### TODO:

```
- [x] Release code.
- [x] Release pre-trained models.
```

