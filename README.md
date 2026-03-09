# LeafInst - Unified Instance Segmentation Network for Fine-Grained Forestry Leaf Phenotype Analysis: A New UAV based Benchmark

## Model Architecture

![LeafInst Model](model.png)
This repository provides the official implementation of **LeafInst**, an instance segmentation framework designed for **forestry leaf phenotyping in complex UAV scenarios**, together with the **Leaf Growth Condition Indicator (LGCI)** computation pipeline.

LeafInst is designed to address the challenges of forestry leaf phenotyping, including:

- Scale variation caused by camera sampling distance  
- Illumination changes from different imaging angles and times  
- Heterogeneous leaf shapes and textures  

To improve instance segmentation performance under these conditions, we introduce:

- **DARH (Dynamic Anomalous Regression Head)** for morphological feature enhancement  
- **TCFU (Top-down Concatenation-decoder Feature Fusion)** to replace traditional **TAFU** in mask feature aggregation  

In addition, we propose **LGCI (Leaf Growth Condition Indicator)**, a quantitative indicator derived from UAV RGB imagery to characterize leaf phenotypic growth conditions for **smart forestry breeding**.

---

## Repository Structure

```text
LeafInst/
│
├── models/                     # Core model components
│   ├── AFPN.py                 # Asymptotic Feature Pyramid Network
│   │                           # Replace FPN.py in MMDetection
│   │
│   ├── TCFU.py                 # TCFU mask feature fusion module
│   │                           # Replace MaskFeatModule in CondInst_head.py
│   │
│   └── DARH.py                 # Dynamic Anomalous Regression Head
│                               # General module head
│
├── lgci/                       # LGCI computation pipeline
│   └── compute_lgci.py         # LGCI indicator calculation and visualization
│
├── configs/                    # Model configuration files
│   └── leafinst_config.py      # Training / inference configuration
│
├── data/                       # Dataset directory (COCO format)
│   ├── images/
│   └── annotations/
│
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── LICENSE                     # License file
```

## Dataset – Poplar-leaf

![dataset](dataset.png)
The Poplar dataset is publicly available on Kaggle:

👉 https://www.kaggle.com/datasets/vvghigh/poplar

To address challenges in forestry leaf phenotyping, we constructed **Poplar-leaf**, a UAV-based benchmark dataset collected in the poplar plantation of Huanghai Seaside National Forest Park, Dongtai City, Jiangsu Province, China (32°51'N, 120°50'E).

Images were acquired using a **DJI M350RTK UAV equipped with a P1 camera**, flying at an altitude of **5 m** along pre-planned routes to capture multi-angle views of tree canopies.

The dataset contains:

- **1,202 labeled images**
- **19,876 leaf instances**
- **1,202 leaf branches**
- Resolution: **1024 × 1024**
- Split: **Train / Val / Test = 8 : 1 : 1**

Pixel-level instance annotations were created using **LabelMe**, supported by the **SAM model** for initial segmentation and refined through manual verification by an 8-person annotation team over 7 days.

To improve model robustness, two data augmentation strategies were applied:

- Random horizontal flip (p = 0.5)
- Random illumination adjustment (0.8–1.1)

Additionally, we provide **40.8 GB of unlabeled UAV imagery**, covering nearly all young poplar trees in the study area, which can be used for **self-supervised learning and transfer learning tasks**.

Poplar-leaf is the **first large-scale UAV-based benchmark dataset for forestry leaf instance segmentation**, capturing natural leaf morphology under diverse illumination and viewing conditions. It provides valuable resources for research in **smart forestry breeding, vegetation phenotyping, and AI-based forest monitoring**.
