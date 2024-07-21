# Cross-View Mutual Learning for Semi-Supervised Medical Image Segmentation (ACM'MM 2024 Poster)
> **Authors:**
Song Wu, Xiaoyu Wei, Xinyue Chen, Yazhou Ren, Jing. He, Xiaorong Pu.

Official code and datas for "Cross-View Mutual Learning for Semi-Supervised Medical Image Segmentation". (ACM'MM 2024)

## 1. Requirements
This repository is based on PyTorch 1.9.1, CUDA 11.6 and Python 3.9.15. All experiments in our paper were conducted on an NVIDIA GeForce RTX 3090 GPU with an identical experimental setting. You should *pip install* some packages for reproducing our experiments:

- scikit-image

- scipy

- tensorboardX

- nibabel

- medpy 

- h5py

- numpy==1.23 (the version >1.24 may cause conflicts with medpy)

## 2. Workflow of CML

<img src="https://github.com/SongwuJob/CML/blob/main/figure/workflow.png"  width="900" />

Overall pipeline of CML in the co-training framework. We first apply the CutMix operation to modify the inputs and supervisory signals to conduct the supervision objective $L_{sup}$. Specifically, for labeled data, we conduct the same CutMix to generate mixed labels $Y^l_{mix}$, which are used to supervise the predictions of $X^l_{mix}$. For unlabeled data, we construct heterogeneous supervisory signals $Y^u_{mix,0}$ and $Y^u_{mix,1}$, obtained by the combination of two subnets' pseudo-labels, to supervise the predictions of $X^u_{mix}$. Note that $\mathcal{L}_{dis}$ is a conflict-based unsupervised loss, aiming to learn distinct features from the same input.

## 3. Usage
We provide `code`, and `data` for LA and ACDC datasets.

To train a model,
``` 
python CML_LA_train.py
python CML_ACDC_train.py
``` 

To test your trained model, and get the final performance,
``` 
python test_LA.py
python test_ACDC.py
```

## 4. Datasets

Data could be got at [LA](https://github.com/yulequan/UA-MT/tree/master/data) and [ACDC](https://github.com/HiLab-git/SSL4MIS/tree/master/data/ACDC).

Particularly, we provide the complete LA and ACDC datasets in [cloud](https://pan.baidu.com/s/1r_0Oh3go_sArFvLxt3QiZg) with **key**: data. You can download directly, and move them to the folder `data`.

## 5. Acknowledgements

Our code is modified from [URPC](https://github.com/HiLab-git/SSL4MIS), [SS-Net](https://github.com/ycwu1997/SS-Net) and [BCP](https://github.com/DeepMed-Lab-ECNU/BCP). Thanks to these authors for their valuable work.
