# CT-Bound: Fast Boundary Estimation From Noisy Images Via Hybrid Convolution and Transformer Neural Networks

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

*Wei Xu, Junjie Luo, and Qi Guo*

Elmore Family School of Electrical and Computer Engineering, Purdue University

Contact: xu1639@purdue.edu

<a href="https://arxiv.org/abs/2403.16494" title="arXiv">**arXiv**</a> | <a href="https://drive.google.com/drive/folders/19TFgtBi1XZiea0ilWVbKvpalh4g7k8ZH?usp=drive_link" title="CT-Bound datasets">**Dataset & Pretrained weights**</a> | <a href="https://youtu.be/MQAExIUfstw" title="CT-Bound video demo">**Real-time video processing demo**</a>

CT-Bound is a fast boundary estimation method for noisy images using a hybrid Convolution and Transformer neural network. The proposed architecture decomposes boundary estimation into two tasks: local detection and global regularization of image boundaries. 

![Qualitative comparison](/pic/comparison.png "Qualitative comparison")

## 1 Usage

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

### 1.1 Training

To train the initialization stage, run

    python init_training.py

To train the refinement stage, run

    python ref_training.py

### 1.2 Testing and evaluation

#### 1.2.1 MS COCO

To investigate the performance of whole pipeline with the MS COCO testing set, run

    python ct_bound.py

#### 1.2.2 BSDS500 and NYUDv2

To evaluate with BSDS500 or NYUDv2 testing set on photo level of $n$ ($\alpha_{\text{test}}=n$), run

    python ct_bound.py --data_path './dataset/eval/' --eval BSDS500 --eval_alpha n

or

    python ct_bound.py --data_path './dataset/eval/' --eval NYUDv2 --eval_alpha n

To calculate the ODS F1-score, run

    cd eval
    matlab -nodisplay -nodesktop -nosplash -r eval_bsds.m

or

    cd eval
    matlab -nodisplay -nodesktop -nosplash -r  eval_nyud.m

Note that if you want to calculate the metrics for color maps, add ``--metrics True`` parameter with the calling command above. 

## 3 Citation

```
@article{xu2024ctbound,
  title={CT-Bound: Fast Boundary Estimation From Noisy Images Via Hybrid Convolution and Transformer Neural Networks}, 
  author={Wei Xu and Junjie Luo and Qi Guo},
  journal={arXiv preprint arXiv:2403.16494},
  year={2024}
}
```

Some of the code is borrowed from <a href="https://github.com/dorverbin/fieldofjunctions/tree/main" title="fieldofjunctions">FoJ</a>.
