# Official implementation for DeCap
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
## Pretrained models
Train on coco captions: [model_coco](https://drive.google.com/file/d/1EFI0aujIWBr3dTC_a2hdoV4QJenAlEWU/view?usp=share_link)

Train on CC3M: Soon
## Citation
```
@inproceedings{lidecap,
  title={DeCap: Decoding CLIP Latents for Zero-Shot Captioning via Text-Only Training},
  author={Li, Wei and Zhu, Linchao and Wen, Longyin and Yang, Yi},
  booktitle={The Eleventh International Conference on Learning Representations}
}
```

## Acknowledgments
This repository is heavily based on [ClipCap](https://github.com/rmokady/CLIP_prefix_caption).
For training we used the data of [COCO dataset](https://cocodataset.org/#home) and [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/).