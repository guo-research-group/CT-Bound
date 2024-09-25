# CT-Bound: Robust Boundary Detection From Noisy Images Via Hybrid Convolution and Transformer Neural Networks

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

*Wei Xu, Junjie Luo, and [Qi Guo](https://qiguo.org)*

Elmore Family School of Electrical and Computer Engineering, Purdue University

Contact: xu1639@purdue.edu

<a href="https://arxiv.org/abs/2403.16494" title="arXiv">**arXiv**</a> | <a href="https://drive.google.com/drive/folders/19TFgtBi1XZiea0ilWVbKvpalh4g7k8ZH?usp=drive_link" title="CT-Bound datasets">**Dataset & Pretrained weights**</a> | <a href="https://youtu.be/MQAExIUfstw" title="CT-Bound video demo">**Real-time video processing demo**</a>

**Content**

- [0 Introduction](#0-introduction)
- [1 Usage](#1-usage)
  * [1.1 Quick start](#11-quick-start)
  * [1.2 Training](#12-training)
  * [1.3 Testing and evaluation](#13-testing-and-evaluation)
    + [1.3.1 MS COCO](#131-ms-coco)
    + [1.3.2 BSDS500 and NYUDv2](#132-bsds500-and-nyudv2)
- [2 Results](#2-results)
- [3 Citation](#3-citation)

## 0 Introduction

CT-Bound is a robust and fast boundary estimation method for noisy images using a hybrid Convolution and Transformer neural network. The proposed architecture decomposes boundary estimation into two tasks: local detection and global regularization of image boundaries. 

![Qualitative comparison](/pic/comparison.png "Qualitative comparison")

## 1 Usage

### 1.1 Quick start

To run the code with conda, please follow the prompts below. 
```
git clone https://github.com/guo-research-group/CT-Bound.git
cd CT-Bound
conda create -n ct_bound python=3.9
conda activate ct_bound
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install -r requirements.txt
```

After downloading and unzipping the dataset and pretrained weights, the full folder content is shown below. 

```
CT-Bound
│   requirements.txt
│   ct_bound.py
│   init_training.py
│   ref_training.py
│   ...
│
└───eval
│   │   eval_bsds.m
│   │   eval_nyud.m
│   │   ...
│
└───dataset
    │   best_ran_pretrained_init.pth
    │   best_ran_pretrained_ref.pth
    │
    └───initialization
    │   │   ...
    │
    └───refinement
    │   │   ...
    │
    └───eval
        │
        └───BSDS500
        │   │   ...
        │
        └───NYUDv2
            │   ...
```

### 1.2 Training

To train the initialization stage, run

    python init_training.py

To train the refinement stage, run

    python ref_training.py

### 1.3 Testing and evaluation

#### 1.3.1 MS COCO

To investigate the performance of whole pipeline with the MS COCO testing set, run

    python ct_bound.py

#### 1.3.2 BSDS500 and NYUDv2

To evaluate with BSDS500 or NYUDv2 testing set on photo level of $n$ ($\alpha_{\text{test}}=n$), run

    python ct_bound.py --data_path './dataset/eval/' --eval 'BSDS500' --eval_alpha n

or

    python ct_bound.py --data_path './dataset/eval/' --eval 'NYUDv2' --eval_alpha n

To calculate the ODS F1-score, run

    cd eval
    matlab -nodisplay -nodesktop -nosplash -r eval_bsds.m

or

    cd eval
    matlab -nodisplay -nodesktop -nosplash -r eval_nyud.m

Note that if you want to calculate the metrics for color maps, add ``--metrics True`` parameter with the calling command above. 

## 2 Results

The ODS F1-score can be found in the table below. 

<table>
    <tr>
        <th style="text-align: center" rowspan="2">Dataset</th>
        <th style="text-align: center" colspan="4">Photon level \alpha</th>
    </tr>
    <tr>
        <th style="text-align: center">2</th>
        <th style="text-align: center">4</th>
        <th style="text-align: center">6</th>
        <th style="text-align: center">8</th>
    </tr>
    <tr>
        <td style="text-align: center">BSDS500</td>
        <td style="text-align: center">0.541</td>
        <td style="text-align: center">0.627</td>
        <td style="text-align: center">0.640</td>
        <td style="text-align: center">0.633</td>
    </tr>
    <tr>
        <td style="text-align: center">NYUDv2</td>
        <td style="text-align: center">0.552</td>
        <td style="text-align: center">0.633</td>
        <td style="text-align: center">0.646</td>
        <td style="text-align: center">0.647</td>
    </tr>
</table>

## 3 Citation

```
@article{xu2024ctbound,
  title={CT-Bound: Robust Boundary Detection From Noisy Images Via Hybrid Convolution and Transformer Neural Networks}, 
  author={Wei Xu and Junjie Luo and Qi Guo},
  journal={arXiv preprint arXiv:2403.16494},
  year={2024}
}
```

Some of the code is borrowed from <a href="https://github.com/dorverbin/fieldofjunctions/tree/main" title="fieldofjunctions">FoJ</a>.
