### Getting started on SPViT-Swin:

#### Installation and data preparation

- First, you can install the required environments as illustrated in the [Swin](https://github.com/microsoft/Swin-Transformer/blob/main/get_started.md) repository or follow the instructions below:

  ```bash
  # Create virtual env
  conda create -n spvit-swin python=3.7 -y
  conda activate spvit-swin
  
  # Install PyTorch
  conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
  pip install timm==0.3.2
  
  # Install Apex
  git clone https://github.com/NVIDIA/apex
  cd apex
  pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
  
  # Install other requirements:
  pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8
  ```

- Next, install some other dependencies that are required by SPViT:

  ```bash
  pip install tensorboardX tensorboard
  ```

- Please refer to the [Swin](https://github.com/microsoft/Swin-Transformer/blob/main/get_started.md) repository to prepare the standard ImageNet dataset, then link the ImageNet dataset under the `dataset`folder:

  ```bash
  $ tree dataset
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

- Since we provide training scripts for three Swin models: Swin-T, Swin-S and Swin-B, please download the corresponding three pre-trained models from the [Swin](https://github.com/microsoft/Swin-Transformer/blob/main/get_started.md) repository as well.

- Next, move the downloaded pre-trained models into the following file structure:

  ```bash
  $ tree model
  ├── swin_base_patch4_window7_224.pth
  ├── swin_small_patch4_window7_224.pth
  ├── swin_tiny_patch4_window7_224.pth
  ```

- Note that do not change the filenames for the pre-trained models as we hard-coded these filenames when tailoring and loading the pre-trained models. Feel free to modify the hard-coded parts when pruning from other pre-trained models.

#### Searching

To search architectures with SPViT-Swin-T, run:

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 3132 main_pruning.py --cfg configs/spvit_swin_tn_l01_t100_search.yaml --resume model/swin_tiny_patch4_window7_224.pth
```

To search architectures with SPViT-Swin-S, run:

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 3132 main_pruning.py --cfg configs/spvit_swin_sm_l04_t55_search.yaml --resume model/swin_small_patch4_window7_224.pth
```

To search architectures with SPViT-Swin-B, run:

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 3132 main_pruning.py --cfg configs/spvit_swin_bs_l01_t100_search.yaml --resume model/swin_base_patch4_window7_224.pth
```

#### Fine-tuning

You can start fine-tuning from either your own searched architectures or from our provided architectures by modifying and assigning the MSA indicators in `assigned_indicators` and the FFN indicators in `searching_model`.

To fine-tune architectures searched by SPViT-Swin-T, run:

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 3132 main_pruning.py --cfg configs/spvit_swin_tn_l01_t100_ft.yaml --resume model/swin_tiny_patch4_window7_224.pth
```

To search architectures with SPViT-Swin-S, run:

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 3132 main_pruning.py --cfg configs/spvit_swin_sm_l04_t55_ft.yaml --resume model/swin_small_patch4_window7_224.pth
```

To search architectures with SPViT-Swin-B, run:

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 3132 main_pruning.py --cfg configs/spvit_swin_bs_l01_t100_ft.yaml --resume model/swin_base_patch4_window7_224.pth
```

#### TODO:

```
- [x] Release code.
- [ ] Release pre-trained models.
```
