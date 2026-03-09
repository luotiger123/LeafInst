# LeafInst: Instance Segmentation for Forestry Leaf Phenotyping with LGCI Evaluation

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
