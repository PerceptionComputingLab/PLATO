# PLATO
[Knowledge-based Systems 2026] Official pytorch implementation of paper "PLATO: ProbabiListic hierArchical mulTi-head mOdel for Plug-and-play Ambiguous Medical Image Segmentation"

## Introduction
Ambiguous medical image segmentation (AMIS) is crucial for computer-aided diagnosis. Current AMIS approaches generally require training specialized models from scratch, which neglects the discriminative capabilities of existing deterministic segmentation models. To address this, we propose PLATO (ProbabiListic hierArchical mulTi-head mOdel ), the first plug-and-play AMIS framework that integrates with arbitrary deterministic segmentation backbones while introducing ambiguity in a fully model-agnostic manner. PLATO features a novel Probabilistic Hierarchical Multi-head Structure (PHMS) that captures inherent ambiguity by hierarchically modeling local ambiguity and global consistency, generating diverse yet anatomically plausible predictions. Additionally, we introduce an Evidence-based Belief Assignment (EBA) module to manage uncertainty entanglement through joint modeling of aleatoric and epistemic uncertainty via Dirichlet-based exponential evidential deep learning. EBA provides a unified, disentangled uncertainty quantification framework that is reliable and interpretable. \hl{Extensive experiments on three public datasets demonstrate that PLATO outperforms existing AMIS methods. Specifically, our PLATO achieves relative improvement compared with existing methods by a maximum of $43.1\%$ on GED$_{\text{100}}$ (PhC Dataset), $60.78\%$ on Brier Score and $0.49\%$ on HM-IoU$_{\text{32}}$ (LIDC Dataset), $26.2\%$ on ECE (ISIC Subset) following commonly used experimental settings and random seed=230.} The proposed PLATO establishes a practical and generalizable plug-and-play solution for ambiguous medical image segmentation. The code will be available.

![Overview of proposed PLATO](/fig/main.png)

## Requirements
You can build the dependencies by executing the following command.
```
conda env create -f eviroment.yml
```

## Datasets
Three public datasets: LIDC-IDRI, ISIC Subset and PhC-U373 are implemented in this work. You can download the datasets from the following links:
- [LIDC-IDRI Dataset](https://drive.google.com/drive/folders/1xKfKCQo8qa6SAr3u7qWNtQjIphIrvmd5) as preprocessed by @Stefan Knegt
- [ISIC Subset](https://drive.google.com/file/d/1m7FdNldGqGyqw2L9GX8HDrHDId3kExtH/view?usp=sharing) as preprocessed by @killanzepf
- [PhC-U373 Dataset](https://drive.google.com/file/d/1TASYQekGtqZUe3VaC4HcbdaVGngSnXcb/view?usp=sharing) as preprocessed by @killanzepf

Please modify the dataset paths accordingly in `metadata_managr.py`

## Training
- Step 1: Train the segmentation backbone (Attention U-Net as an example in this repo):
```
python train_atten_unet.py --what LIDC --epochs 1000 --batchsize 64 
```
- Step 2: Train PLATO wth the segmentation backbone frozen:
```
python train_atten_unet_MELT.py --what LIDC --epochs 200 --batchsize 64 
```
## Testing
```
python test_atten_unet_MELT.py --what LIDC 
```

## Acknowledgements
- We thank [@killanzepf](https://github.com/kilianzepf/conditioned_uncertain_segmentation) for the preprocessed dataset and PLATO baseline.
- We thank [@Stefan Knegt](https://github.com/stefanknegt/Probabilistic-Unet-Pytorch) for the preprocessed dataset.

## Citations
If you find this work helpful, please cite:
```
@article{li2026PLATO,
  title={PLATO: ProbabiListic hierArchical mulTi-head mOdel for Plug-and-play Ambiguous Medical Image Segmentation},
  author={Li, Xiangyu and Li, Fanding and Yuan, Yongfeng and Dong, Suyu and Wang, Kuanquan and Shen, Yi and Wang, Guohua and Luo, Gongning and Li, Shuo},
  journal={Knowledge-based Systems},
  pages={115746},
  year={2025}
}
```
