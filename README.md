# MFCFNet: Multi-feature collaborative fusion network with deep supervision for sar ship classification
This is the implementation of our paper: Multi-feature collaborative fusion network with deep supervision for sar ship classification (TGRS 2023). 
<div align=center><img  src="https://github.com/StuZheng/MFCFNet/blob/master/fig/MFCFNet.png"/></div>

(a)–(g) Seven categories of SAR ship images in the FUSAR-Ship dataset. (a) Container, (b) general cargo, (c) fishing, (d) tanker, (e) bulk, (f) other
cargo, and (g) others. The first row is the original image and the other rows are corresponding handcrafted feature visualizations.
<div align=center><img  src="https://github.com/StuZheng/MFCFNet/blob/master/fig/Hand_Feature.png"/></div>

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
