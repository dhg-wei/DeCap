# DeCap
DeCap: Decoding CLIP Latents for Zero-Shot Captioning via Text-Only Training

Published at ICLR 2023 

Paper link: [DeCap](https://openreview.net/pdf?id=Lt8bMlhiwx2)

## Data
Download [coco_train](https://drive.google.com/file/d/1k4LlhgwnvpkUlzQjtTomnDFvlkboTxOH/view?usp=share_link) to `data`.
Download [cc3m_train](https://drive.google.com/file/d/1-xfOLJasBTqTrSnsyAncKSfsjSSN5RTH/view?usp=share_link) to `data`.
## Training
```
./train_coco.sh
```
or 
```
./train_cc3m.sh
```
## Inferece
See `inference_decap.ipynb`.
## Citation
```
@inproceedings{
li2023decap,
title={DeCap: Decoding {CLIP} Latents for Zero-Shot Captioning via Text-Only Training},
author={Wei Li and Linchao Zhu and Longyin Wen and Yi Yang},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=Lt8bMlhiwx2}
}
```
## Acknowledgments
This repository is heavily based on [ClipCap](https://github.com/rmokady/CLIP_prefix_caption).
For training we used the data of [COCO dataset](https://cocodataset.org/#home) and [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/).