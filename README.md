# AMSP-UOD
```
The 38th Annual AAAI Conference on Artificial Intelligence (AAAI), 2024
```
This is an official PyTorch implementation of paper [AMSP-UOD: When Vortex Convolution and Stochastic Perturbation Meet Underwater Object Detection](https://arxiv.org/abs/2308.11918)
![fig1 (5)](https://github.com/Hikari0608/Underwater-object-detection/assets/88725956/b719e189-a34e-474f-af61-f9c1cad7220b)

## Quick Start

This code repository includes the base run code for AMSP-UOD, see folder `./weights` for the weights file, and see folder `./result` for the PR curve graph. There is some of the urpc test data in folder `./urpc`, and you can get the recognition results using `./det.sh`

### 1. Deploy Conda environment
```Command Line
conda create -n AMSP_UOD python==3.10
```

### 2. Install package dependencies
```Command Line
pip install -r requirements.txt
```

### 3. Train Model (Optional, requires Datasets and Cuda)
```Command Line
conda activate AMSP_UOD
./train.sh 0
```

### 4. Test Model (Optional, requires Datasets and Cuda)
Note that when testing model performance, change the val option in urpc.yaml from self-divided data to urpc's B-list data.
```Command Line
conda activate AMSP_UOD
./val.sh
```

### 5. Dectet
```Command Line
conda activate AMSP_UOD
./det.sh
```

### Showcase

#### URPC
![img1](./result/Traditional-NMS/URPC-Ours_AMSP_UOD.png)
#### RUOD
![img2](./result/Traditional-NMS/RUOD-Ours_AMSP_UOD.png)

For more details check out `./result` folder, we give the experimental result plots for some of the ablation experiments.

## Cite
You can cite our work in the following format:

### arXiv
```bibtex
@article{zhou2023amsp,
  title={AMSP-UOD: When Vortex Convolution and Stochastic Perturbation Meet Underwater Object Detection},
  author={Zhou, Jingchun and He, Zongxin and Lam, Kin-Man and Wang, Yudong and Zhang, Weishi and Guo, ChunLe and Li, Chongyi},
  journal={arXiv preprint arXiv:2308.11918},
  year={2023}
}
```

### AAAI-24
```bibtex
Accepted, will update shortly.
```
