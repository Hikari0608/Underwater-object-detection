# [Research for Underwater Object Detection] Balancing Real-Time and Accuracy in Resource-Constrained Underwater Scenarios

The TRT weights used are publicly available and the training code is being processed. ☺

## Directory Structure (Continuously updated)

```
Spatial-Residual
├── BSR5-DETR
│   └── weights
│       ├── rtdetr-bsr5-1024.engine (Ours)
│       ├── rtdetr-bsr5-512.engine (Ours)
│       ├── rtdetr-r101.engine
│       └── rtdetr-r50.engine
├── BSR5-YOLO
│   └── weights
│       ├── bsr5-l.engine (Ours)
│       ├── bsr5-m.engine (Ours)
│       ├── bsr5-s.engine (Ours)
│       ├── yolov8l.engine
│       ├── yolov8l_ruod.engine (MMYOLO)
│       ├── yolov8m.engine
│       ├── yolov8m_ruod.engine (MMYOLO)
│       ├── yolov8s.engine
│       └── yolov8s_ruod.engine (MMYOLO)
├── datas
│   ├── ruod.yaml
│   └── urpcB.yaml
└── README.md
```

Each directory includes detailed documentation and relevant materials for the corresponding project.

## API Installation Guide

For performance validation or image prediction tasks, you may utilize APIs from [Ultralytics](https://github.com/ultralytics/ultralytics). Follow the steps below:

### Installation (Only for Val/Predict)

Ensure your Python environment is properly set up, then run the following command to install necessary dependencies:

```bash
conda create -n uod python=3.8 -y
conda activate uod
pip install ultralytics
```

### Usage

For specific instructions on how to use these APIs for performance validation or image prediction, please refer to the [official Ultralytics documentation](https://github.com/ultralytics/ultralytics).


#### Val
```bash
conda activate uod
yolo val model=bsr5-l.engine data=ruod.yaml
yolo val model=rtdetr-bsr5-1024.engine data=ruod.yaml
```

the output will like:
```
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 4200/4200 [00:49<00:00, 84.76it/s] 
                   all       4200      22969      0.861       0.82      0.879      0.661
           holothurian       4200       2382      0.847      0.778      0.844      0.546
               echinus       4200       3425      0.884      0.857      0.916       0.57
               scallop       4200       2373      0.844       0.75      0.852      0.563
              starfish       4200       2373      0.856      0.867      0.915      0.598
                  fish       4200       4001      0.808      0.693      0.789      0.569
                corals       4200       2751      0.761      0.711      0.765      0.594
                 diver       4200       1997      0.917      0.914      0.951      0.774
            cuttlefish       4200       1607      0.964       0.96      0.976      0.874
                turtle       4200       1296      0.952       0.95      0.975      0.874
             jellyfish       4200        764      0.779      0.721      0.807      0.651
Speed: 0.4ms preprocess, 5.3ms inference, 0.0ms loss, 1.1ms postprocess per image
```

#### Predict
```bash
conda activate uod
yolo predict model=bsr5-l.engine source=<img-path>
yolo predict model=rtdetr-bsr5-1024.engine source=<img-path>
```

## Citation

If you find any part of this repository useful for your work, please consider citing the relevant papers as follows:

### AAAI-2024

```bibtex
@inproceedings{amsp,
  title={AMSP-UOD: When vortex convolution and stochastic perturbation meet underwater object detection},
  author={Zhou, Jingchun and He, Zongxin and Lam, Kin-Man and Wang, Yudong and Zhang, Weishi and Guo, Chunle and Li, Chongyi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={7},
  pages={7659--7667},
  year={2024}
}
```

### TPAMI-2025

```bibtex
@ARTICLE{spatial,
  author={Zhou, Jingchun and He, Zongxin and Zhang, Dehuan and Liu, Siyuan and Fu, Xianping and Li, Xuelong},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Spatial Residual for Underwater Object Detection}, 
  year={2025},
  volume={},
  number={},
  pages={},
  doi={10.1109/TPAMI.2025.3548652}}
```

Should you have any questions or need further assistance, feel free to reach out through GitHub Issues.
