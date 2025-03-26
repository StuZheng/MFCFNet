# MFCFNet: Multi-feature collaborative fusion network with deep supervision for sar ship classification
This is the implementation of our paper: Multi-feature collaborative fusion network with deep supervision for sar ship classification (TGRS 2023). 
<div align=center><img  src="https://github.com/StuZheng/MFCFNet/blob/master/fig/MFCFNet.png"/></div>

(a)–(g) Seven categories of SAR ship images in the FUSAR-Ship dataset. (a) Container, (b) general cargo, (c) fishing, (d) tanker, (e) bulk, (f) other
cargo, and (g) others. The first row is the original image and the other rows are corresponding handcrafted feature visualizations.
<div align=center><img  src="https://github.com/StuZheng/MFCFNet/blob/master/fig/Hand_Feature.png"/></div>

## Abstract

Multifeature synthetic aperture radar (SAR) ship classification aims to build models that can process, correlate, and fuse information from both handcrafted and deep features. Although handcrafted features provide rich expert knowledge, current fusion methods inadequately explore the relatively significant role of handcrafted features in conjunction with deep features, the imbalances in feature contributions, and the cooperative ways in which features learn. In this article, we propose a novel multifeature collaborative fusion network with deep supervision (MFCFNet) to effectively fuse handcrafted features and deep features for SAR ship classification tasks. Specifically, our framework mainly includes two types of feature extraction branches, a knowledge supervision and collaboration module (KSCM) and a feature fusion and contribution assignment module (FFCA). The former module improves the quality of the feature maps learned by each branch through auxiliary feature supervision and introduces a synergy loss to facilitate the interaction of information between deep features and handcrafted features. The latter module utilizes an attention mechanism to adaptively balance the importance among various features and assign the corresponding feature contributions to the total loss function based on the generated feature weights. We conducted extensive experimental and ablation studies on two public datasets, OpenSARShip-1.0 and FUSAR-Ship, and the results show that MFCFNet is effective and outperforms single deep feature and multifeature models based on previous internal FC layer and terminal FC layer fusion. Furthermore, our proposed MFCFNet exhibits better performance than the current state-of-the-art methods.


## 📦 Environment

```
conda create -n xxxx
conda install --yes --file requirements.txt
```

## 💡 RUN

```
python hand_MultiCNN.py
```

## 🧑🏻‍💻 Bugs or Questions?

If you have any questions related to the code or the paper, feel free to email Junbao (zhenghao@csu.edu.cn). If you
encounter any problems when using the code, or want to report a bug, you can open an issue.


## 📝 Citation

If you find MFCFNet useful for your research, please consider citing our paper:

```
@article{zheng2023multifeature,
  title={Multifeature collaborative fusion network with deep supervision for SAR ship classification},
  author={Zheng, Hao and Hu, Zhigang and Yang, Liu and Xu, Aikun and Zheng, Meiguang and Zhang, Ce and Li, Keqin},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={61},
  pages={1--14},
  year={2023},
  publisher={IEEE}
}
```
