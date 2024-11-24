# RGB-Event Fusion for Moving Object Detection in Autonomous Driving (ICRA'23)

This repository is for the paper **RGB-Event Fusion for Moving Object Detection in Autonomous Driving**, by
[Zhuyun Zhou](https://scholar.google.com/citations?user=sXolUXMAAAAJ&hl=en&oi=ao),
[Zongwei Wu](https://scholar.google.com/citations?user=3QSALjX498QC&hl=en&oi=ao),
[Rémi Boutteau](https://scholar.google.com/citations?user=U-SrcPkAAAAJ&hl=en&oi=ao),
[Fan Yang](https://scholar.google.com/citations?user=GNQHje8AAAAJ&hl=en&oi=ao),
[Cédric Demonceaux](https://scholar.google.com/citations?user=CCvaUR4AAAAJ&hl=en&oi=ao),
[Dominique Ginhac](https://scholar.google.com/citations?user=fkdCT5kAAAAJ&hl=en&oi=ao).

PDF version of the paper is available [here](https://arxiv.org/abs/2209.08323).

Dataset ***DSEC-MOD*** **: ***DSEC*** - ***M***oving ***O***bject ***D***etection** can be found [here](#dataset).



## Contents

1. [Abstract](#abstract)
2. [News](#news)
3. [Citation](#citation)
4. [Dataset](#dataset)
5. [Pre-trained Weights](#pre-trained-weights)
6. [Installation](#installation)



## Abstract

![Graphical Abstract](https://user-images.githubusercontent.com/114169266/231510781-d7a185fb-a121-42e7-a0a7-2b12c8d15cf7.png)



Moving Object Detection (MOD) is a critical vision task for successfully achieving safe autonomous driving. Despite plausible results of deep learning methods, most existing approaches are only frame-based and may fail to reach reasonable performance when dealing with dynamic traffic participants. Recent advances in sensor technologies, especially the Event camera, can naturally complement the conventional camera approach to better model moving objects. However, event-based works often adopt a pre-defined time window for event representation, and simply integrate it to estimate image intensities from events, neglecting much of the rich temporal information from the available asynchronous events. Therefore, from a new perspective, we propose RENet, a novel RGB-Event fusion Network, that jointly exploits the two complementary modalities to achieve more robust MOD under challenging scenarios for autonomous driving. Specifically, we first design a temporal multi-scale aggregation module to fully leverage event frames from both the RGB exposure time and larger intervals. Then we introduce a bi-directional fusion module to attentively calibrate and fuse multi-modal features. To evaluate the performance of our network, we carefully select and annotate a sub-MOD dataset from the commonly used DSEC dataset. Extensive experiments demonstrate that our proposed method performs significantly better than the state-of-the-art RGB-Event fusion alternatives.



## News

* Mar. 17, 2023: Code of our ***RENet*** **: ***R***GB-***E***vent fusion ***Net***work** is released.

* Mar. 14, 2023: Dataset ***DSEC-MOD*** **: ***DSEC*** - ***M***oving ***O***bject ***D***etection** is released.



## Citation

If you use any of this code or the dataset DSEC-MOD in your research, please cite the following work:



```BibTeX
@inproceedings{zhou2023rgb,
  title={Rgb-event fusion for moving object detection in autonomous driving},
  author={Zhou, Zhuyun and Wu, Zongwei and Boutteau, R{\'e}mi and Yang, Fan and Demonceaux, C{\'e}dric and Ginhac, Dominique},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={7808--7815},
  year={2023},
  organization={IEEE}
}
```







## Dataset

***DSEC-MOD*** **: ***DSEC*** - ***M***oving ***O***bject ***D***etection** can be downloaded here: [Training](https://drive.google.com/file/d/1Yi1sdp3OIHSg3EplwJsEFHNF5-_dmBO5/view?usp=sharing) and [Testing](https://drive.google.com/file/d/1A7YcmY4s0X2Db2gi_Xt1rhNt-8mRXJ2f/view?usp=sharing).

In total, our DSEC-MOD dataset contains 16 sequences (13314 frames), with 11 sequences (10495 frames) for training and 5 other sequences (2819 frames) for testing.

In each sequence:
* `gt_bb`: ground truth bounding boxes of moving objects;
* `rgb_calib`: RGB frames calibrated to the event-based coordinates, so that RGB and event maps have the same field of view and the same resolution;
* `events`: event data from left sensor.

The format should be:
```
└── DSEC_MOD
    ├── training
    │   ├── zurich_city_00_a
    │   │   ├── gt_bb
    │   │   │   ├── 000001.txt
    │   │   │   └── ...
    │   │   ├── rgb_calib
    │   │   │   ├── 000001.png
    │   │   │   └── ...
    │   │   └── events  
    │   │       └── left
    │   │           ├── events.h5
    │   │           └── rectify_map.h5
    │   └── ...
    └── testing
        ├── zurich_city_13_a
        │   └── ...
        └── ... 
```



### Parent Dataset: DSEC

DSEC is available here: [ https://dsec.ifi.uzh.ch](https://dsec.ifi.uzh.ch).

Details can be found in the paper [ DSEC: A Stereo Event Camera Dataset for Driving Scenarios](https://rpg.ifi.uzh.ch/docs/RAL21_DSEC.pdf).



### Sister Dataset: DSEC-MOS

DSEC-MOS is available here: [ https://github.com/ZZY-Zhou/DSEC-MOS](https://github.com/ZZY-Zhou/DSEC-MOS).

Details can be found in the paper [ DSEC-MOS: Segment Any Moving Object with Moving Ego Vehicle](https://arxiv.org/abs/2305.00126).



## Pre-trained Weights

Our pre-trained weights for our RENet can be downloaded [here](https://drive.google.com/file/d/175fW9v4BevwD-U--dJFzq0aYlXzp6sHm/view?usp=sharing).

To get the same experimental results as in our paper, Events should be pre-processed by 3 temporal scales ([15ms](https://drive.google.com/file/d/1yS2oTtAxbOC8Z_IJ67ttI-aYq5EzSYOO/view?usp=sharing), [30ms](https://drive.google.com/file/d/1nRHnQyxQOXo5RA2Gj1_PME7Vm61s-isc/view?usp=sharing), [50ms](https://drive.google.com/file/d/19CX9VyuC2HAqof9xW-UnUttZy7CIFrdH/view?usp=sharing)), details can be found in *Section III-A. E-TMA: Event-based Temporal Multi-scale Aggregation* in our [paper](https://arxiv.org/abs/2209.08323).

Then run the inference:

```
python3 det.py --task stream --model ROOT_OF_MODEL/best_model.pth --inference_dir PATH_TO_INF
```

The following command is to get the frame or video mAP:

```
python3 ACT.py --task TASK_NAME --th THRESHOLD --inference_dir PATH_TO_INF
```

For instance:
```
python3 ACT.py --task frameAP --th 0.5 --inference_dir PATH_TO_INF
```
or
```
python3 ACT.py --task videoAP --th 0.2 --inference_dir PATH_TO_INF
```



The initial pre-trained weights are also available: [ResNet-101](https://drive.google.com/file/d/1PsAJUV7DD6fNSh54Biy9l3bGNRxBn4oB/view?usp=sharing) backbone for RGB stream, and [ResNet-18](https://drive.google.com/file/d/1ccJ1k9G4Dqk0LOwogHJu0n7XUlPAr4N5/view?usp=sharing) backbone for Event stream.



## Installation

1. Clone

```
git clone https://github.com/ZZY-Zhou/RENet
cd RENet
```

2. Create and activate conda environment

```
conda create -n ENV_NAME
conda activate ENV_NAME
```


