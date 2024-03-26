# CT-Bound: Fast Boundary Estimation From Noisy Images Via Hybrid Convolution and Transformer Neural Networks

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

*Wei Xu, Junjie Luo, and Qi Guo*

Elmore Family School of Electrical and Computer Engineering, Purdue University

Contact: xu1639@purdue.edu

CT-Bound (<a href="https://arxiv.org/abs/2403.16494" title="CT-Bound">**paper**</a>) is a fast boundary estimation method for noisy images using a hybrid Convolution and Transformer neural network. The proposed architecture decomposes boundary estimation into two tasks: local detection and global regularization of image boundaries. 

![Qualitative comparison](/pic/comparison.png "Qualitative comparison")

## Usage

Our datasets and the pretrained model can be found <a href="https://drive.google.com/drive/folders/19TFgtBi1XZiea0ilWVbKvpalh4g7k8ZH?usp=drive_link" title="CT-Bound datasets">here</a>. We also have a video demo to show the real time processing <a href="https://youtu.be/MQAExIUfstw" title="CT-Bound video demo">here</a>. 

The folder content is shown below. Please create the ``dataset`` folder and its subfolders and put the datasets into the the corresponding folders. Note that the trained model will be saved in the corresponding subfolders by default. In our implementation, the number of edge parameters in each patch is 3. 

```
CT-Bound
│   environment.yml
│   ct_bound.py
│   init_training.py
│   ref_training.py
│   ...
│
└───dataset
    │
    └───initialization
    │   │   best_run.pth
    │   │   ...
    │
    └───refinement
        │   best_run.pth
        │   ...
```

To train the initialization stage, run

    python init_training.py

To train the refinement stage, run

    python ref_training.py

To investigate the performance of whole pipeline with our the testing set, run

    python ct_bound.py

## Citation

```
@article{xu2024ctbound,
  title={CT-Bound: Fast Boundary Estimation From Noisy Images Via Hybrid Convolution and Transformer Neural Networks}, 
  author={Wei Xu and Junjie Luo and Qi Guo},
  journal={arXiv preprint arXiv:2403.16494},
  year={2024}
}
```

Some of the code is borrowed from <a href="https://github.com/dorverbin/fieldofjunctions/tree/main" title="fieldofjunctions">FoJ</a>.
