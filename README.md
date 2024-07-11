# Integer-Valued Training and Spike-Driven Inference Spiking Neural Network for High-performance and Energy-efficient Object Detection (ECCV2024)

[Xinhao Luo](), [Man Yao](https://scholar.google.com/citations?user=eE4vvp0AAAAJ), [Yuhong Chou](https://scholar.google.com.hk/citations?hl=zh-CN&user=8CpWM4cAAAAJ), [Bo Xu]() and [Guoqi Li](https://scholar.google.com/citations?user=qCfE--MAAAAJ&)

BICLab, Institute of Automation, Chinese Academy of Sciences

---

:rocket:  :rocket:  :rocket: **News**:

- **Jun. 1, 2024**: Accepted as poster in ECCV2024.


TODO:

- [x] Upload codes.
- [x] Upload checkpoints.

## Abstract

Brain-inspired Spiking Neural Networks (SNNs) have bio-plausibility and low-power advantages over Artificial Neural Networks (ANNs). Applications of SNNs are currently limited to simple classification tasks because of their poor performance. In this work, we focus on bridging the performance gap between ANNs and SNNs on object detection. Our design revolves around network architecture and spiking neuron, include:(1)**SpikeYOLO**, We explore suitable architectures in SNNs for handling object detection tasks and propose SpikeYOLO, which simplifies YOLOv8 and incorporates meta SNN blocks. This inspires us that the complex modules in ANN may not be suitable for SNN architecture design. **I-LIF Spiking Neuron**, We propose an I-LIF spiking neuron that combines integer-valued training with spike-driven inference. The former is used to reduce quantization errors in spiking neurons, and the latter is the basis of the low-power nature of SNNs. The proposed method achieves outstanding accuracy with low power consumption on object detection datasets, demonstrating the potential of SNNs in complex vision tasks. On the COCO dataset, we obtain \textbf{66.2\%} mAP@50 and \textbf{48.9\%} mAP@50:95, which is \textbf{+15.0\%} and \textbf{+18.7\%} higher than the prior state-of-the-art SNN, respectively. On the Gen1 dataset, SpikeYOLO is \textbf{+2.5\%} better than ANN models with \textbf{5.7}$\times$ energy efficiency.

![image](figure1.pdf)



For help or issues using this git, please submit a GitHub issue.

For other communications related to this git, please contact `luoxinhao2023@ia.ac.cn` and `manyao@ia.ac.cn`.

## Thanks

Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

[deit](https://github.com/facebookresearch/deit)
