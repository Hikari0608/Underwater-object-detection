# Underwater Object Detection

This repository collects the code, weights, and notes for two underwater object detection lines of work:

- AMSP-UOD, the AAAI-2024 project under `AMSP-UOD/`
- Spatial Residual for Underwater Object Detection, the TPAMI-2025 project under `Spatial-Residual/`

The root README is an index only. Usage, training, validation, and prediction commands live in the corresponding subdirectory README files for each code version.

## Directory Map

| Directory | Method / version | What it contains | Where to read usage |
| --- | --- | --- | --- |
| `AMSP-UOD/original` | AMSP-UOD original implementation, AAAI-2024 | Original PyTorch code, scripts, weights, and results | [AMSP-UOD/original/README.md](AMSP-UOD/original/README.md) |
| `AMSP-UOD/latest` | AMSP-UOD latest packaged version | Latest runtime assets and exported weights | See the files in this directory |
| `Spatial-Residual/BSR5-DETR` | BSR5-DETR, TPAMI-2025 code branch | Training, evaluation, and export code for the RT-DETR-based model | [Spatial-Residual/BSR5-DETR/README.md](Spatial-Residual/BSR5-DETR/README.md) |
| `Spatial-Residual/BSR5-DETR-ultralytics` | BSR5-DETR Ultralytics-compatible version | Ultralytics-style inference and validation assets | See the files in this directory |
| `Spatial-Residual/BSR5-YOLO` | BSR5-YOLO release branch, TPAMI-2025 | Released weights and related materials | See the files in this directory |
| `Spatial-Residual/datas` | Shared datasets | Dataset configuration files such as `ruod.yaml` and `urpcB.yaml` | See the files in this directory |

## Repository Notes

The available TRT weights are published in the release page linked in the project documentation. The corresponding usage examples are intentionally kept in the code-version directories so that each method can maintain its own environment and command set.

For the most up-to-date implementation details, open the README in the target subdirectory first.

## Citation

If you use this repository in your work, please cite the relevant paper for the code path you use.

### AMSP-UOD, AAAI-2024

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

### Spatial Residual, TPAMI-2025

```bibtex
@ARTICLE{spatial,
  title={Spatial residual for underwater object detection},
  author={Zhou, Jingchun and He, Zongxin and Zhang, Dehuan and Liu, Siyuan and Fu, Xianping and Li, Xuelong},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025},
  publisher={IEEE}
}
```

For questions or implementation details, please open the README in the relevant subdirectory or use GitHub Issues.
