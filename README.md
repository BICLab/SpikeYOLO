# Spike-driven Transformer V2: Meta Spiking Neural Network Architecture Inspiring the Design of Next-generation Neuromorphic Chips ([ICLR2024](https://openreview.net/forum?id=1SIBN5Xyw7))

[Man Yao](https://scholar.google.com/citations?user=eE4vvp0AAAAJ), [Jiakui Hu](https://github.com/jkhu29), [Tianxiang Hu](), [Yifan Xu](https://scholar.google.com/citations?hl=zh-CN&user=pbcoTgsAAAAJ), [Zhaokun Zhou](https://scholar.google.com/citations?user=4nz-h1QAAAAJ), [Yonghong Tian](https://scholar.google.com/citations?user=fn6hJx0AAAAJ), [Bo Xu](), [Guoqi Li](https://scholar.google.com/citations?user=qCfE--MAAAAJ&)

BICLab, Institute of Automation, Chinese Academy of Sciences

---

:rocket:  :rocket:  :rocket: **News**:

- **Jan. 16, 2024**: Accepted as poster in ICLR2024.
- **Feb. 15, 2024**: Release the training and inference codes in classification tasks.
- **Apr. 19, 2024**: Release the [pre-trained ckpts and training logs](https://drive.google.com/drive/folders/12JcIRG8BF6JcgPsXIetSS14udtHXeSSx?usp=sharing) of SDT-v2.

TODO:

- [x] Upload train and test scripts.
- [x] Upload checkpoints.

## Abstract

Neuromorphic computing, which exploits Spiking Neural Networks (SNNs) on neuromorphic chips, is a promising energy-efficient alternative to traditional AI. CNN-based SNNs are the current mainstream of neuromorphic computing. By contrast, no neuromorphic chips are designed especially for Transformer-based SNNs, which have just emerged, and their performance is only on par with CNN-based SNNs, offering no distinct advantage. In this work, we propose a general Transformer-based SNN architecture, termed as "Meta-SpikeFormer", whose goals are: (1) **Lower-power**, supports the spike-driven paradigm that there is only sparse addition in the network; (2) **Versatility**, handles various vision tasks; (3) **High-performance**, shows overwhelming performance advantages over CNN-based SNNs; (4) **Meta-architecture**, provides inspiration for future next-generation Transformer-based neuromorphic chip designs. Specifically, we extend the [Spike-driven Transformer](https://github.com/BICLab/Spike-Driven-Transformer) into a meta architecture, and explore the impact of structure, spike-driven self-attention, and skip connection on its performance. On ImageNet-1K, Meta-SpikeFormer achieves **80.0% top-1 accuracy** (55M), surpassing the current state-of-the-art (SOTA) SNN baselines (66M) by 3.7%. This is the first direct training SNN backbone that can simultaneously **supports classification, detection, and segmentation**, obtaining SOTA results in SNNs. Finally, we discuss the inspiration of the meta SNN architecture for neuromorphic chip design.

![V2](./img/300_spike_driven_transformer_v2_me.png)

## Classification

### Requirements

```python3
pytorch >= 2.0.0
cupy
spikingjelly == 0.0.0.0.12
```

### Results on Imagenet-1K

Pre-trained ckpts and training logs of 55M: [here](https://drive.google.com/drive/folders/12JcIRG8BF6JcgPsXIetSS14udtHXeSSx?usp=sharing).

### Train & Test

The hyper-parameters are in `./conf/`.

Train:

```shell
torchrun --standalone --nproc_per_node=8 \
  main_finetune.py \
  --batch_size 128 \
  --blr 6e-4 \
  --warmup_epochs 10 \
  --epochs 200 \
  --model metaspikformer_8_512 \
  --data_path /your/data/path \
  --output_dir outputs/T1 \
  --log_dir outputs/T1 \
  --model_mode ms \
  --dist_eval
```

Finetune:

> Please download caformer_b36_in21_ft1k.pth first following [PoolFormer](https://github.com/sail-sg/poolformer).

```shell
torchrun --standalone --nproc_per_node=8 \
  main_finetune.py \
  --batch_size 24 \
  --blr 2e-5 \
  --warmup_epochs 5 \
  --epochs 50 \
  --model metaspikformer_8_512 \
  --data_path /your/data/path \
  --output_dir outputs/T4 \
  --log_dir outputs/T4 \
  --model_mode ms \
  --dist_eval \
  --finetune /your/ckpt/path \
  --time_steps 4 \
  --kd \
  --teacher_model caformer_b36_in21ft1k \
  --distillation_type hard
```

Test:

```shell
python main_finetune.py --batch_size 128 --model metaspikformer_8_512 --data_path /your/data/path --eval --resume /your/ckpt/path
```

### Data Prepare

ImageNet with the following folder structure, you can extract imagenet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

```shell
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

## Contact Information

```
@inproceedings{
yao2024spikedriven,
title={Spike-driven Transformer V2: Meta Spiking Neural Network Architecture Inspiring the Design of Next-generation Neuromorphic Chips},
author={Man Yao and JiaKui Hu and Tianxiang Hu and Yifan Xu and Zhaokun Zhou and Yonghong Tian and Bo XU and Guoqi Li},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=1SIBN5Xyw7}
}
```

For help or issues using this git, please submit a GitHub issue.

For other communications related to this git, please contact `manyao@ia.ac.cn` and `jkhu29@stu.pku.edu.cn`.

## Thanks

Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

[deit](https://github.com/facebookresearch/deit)
