# CT-Bound: Fast Boundary Estimation From Noisy Images Via Hybrid Convolution and Transformer Neural Networks

*Wei Xu, Junjie Luo, and Qi Guo*

School of Electrical and Computer Engineering, Purdue University

Contact: xu1639@purdue.edu

CT-Bound is a fast boundary estimation method for noisy images using a hybrid Convolution and Transformer neural network. The proposed architecture decomposes boundary estimation into two tasks: local detection and global regularization of image boundaries. 

![Qualitative comparison](/pic/comparison.png "Qualitative comparison")

Our datasets and the pretrained model can be found <a href="https://drive.google.com/drive/folders/19TFgtBi1XZiea0ilWVbKvpalh4g7k8ZH?usp=drive_link" title="CT-Bound datasets">here</a>. We also have a demo video to show the real time processing <a href="place_holder" title="CT-Bound demo video">here</a>. 

The folder content is shown below. Please create the ``dataset`` folder and its subfolder and put the datasets into the the corresponding folders. Note that the trained model will be saved in the corresponding subfolders by default. 

```
CT-Bound
│   environment.yml
│   ct_bound.py
|   init_training.py
|   ref_training.py
|   transformer_origianl.py
|   ...
│
└───dataset
    │
    └───initialization
    |   │   best_run.pth
    |   │   ...
    |
    └───refinement
        |   best_run.pth
        |   ...
```

To train the initialization stage, run

    python init_training.py

To train the refinement stage, run

    python ref_training.py

To investigate the performance of whole pipeline with our the testing set, run

    python ct_bound.py

Some of the code is borrowed from <a href="https://github.com/dorverbin/fieldofjunctions/tree/main" title="fieldofjunctions">FoJ</a>.
