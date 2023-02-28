# DeCap
ICLR 2023 DeCap: Decoding CLIP Latents for Zero-shot Captioning

## Data
Download [coco_train](https://drive.google.com/file/d/1k4LlhgwnvpkUlzQjtTomnDFvlkboTxOH/view?usp=share_link) to `data`.
Download [cc3m_train](https://drive.google.com/file/d/1-xfOLJasBTqTrSnsyAncKSfsjSSN5RTH/view?usp=share_link) to `data`.
## Train
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
Soon
## Acknowledgments
This repository is heavily based on [ClipCap](https://github.com/rmokady/CLIP_prefix_caption).
For training we used the data of [COCO dataset](https://cocodataset.org/#home) and [Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/).