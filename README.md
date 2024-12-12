# R-MeeTo: Rebuild Your Faster Vision Mamba in Minutes

The official implementation of "Faster Vision Mamba is Rebuilt in Minutes via Merged Token Re-training".

> Mingjia Shi<sup></sup>, Yuhao Zhou<sup></sup>, Ruiji Yu<sup></sup>, Zekai Li<sup></sup>, Zhiyuan Liang<sup></sup>, Xuanlei Zhao<sup></sup>, Xiaojiang Peng<sup></sup>, Tanmay Rajpurohit<sup></sup>, Ramakrishna Vedantam<sup></sup>, Wangbo Zhao<sup></sup>, Kai Wang<sup></sup>, Yang You<sup></sup>
>
<!-- > <sup>1</sup>[National University of Singapore](https://www.nus.edu.sg/), <sup>2</sup>[DAMO Academy, Alibaba Group](https://damo.alibaba.com/?language=zh), <sup>3</sup>Hupan Lab, <sup>4</sup>[Tsinghua University](https://www.tsinghua.edu.cn/) -->
>
>  [Paper]()


## Overview
<p align="center">
<img src="./fig/R_MeeTo.png" width=100% height=45%
class="center">

Figure: Analysis’ sketch: Mamba is sensitive to token reduction. Experiments about **i.** token reduction are conducted with DeiT-S
(Transformer) and Vim-S (Mamba) on ImageNet-1K. The reduction ratios in the experiment about **ii.** shuffled tokens are 0.14 for
Vim-Ti and 0.31 for Vim-S/Vim-B. Shuffle strategy is odd-even shuffle: [0,1,2,3]→[0,2],
[1,3]→[0,2,1,3]. The empirical results of I(X;Y), the mutual information between inputs X and outputs Y of the Attention Block and
SSM, are measured by MINE on the middle layers of DeiT-S and Vim-S (7-th/12 layers and the 14-th/24 layers respectively.) See this implementation repo of [MINE](https://github.com/BDeMo/MINE_SSM_Attention).   

> **Abstract:**
Vision Mamba (e.g., Vim) has successfully been integrated into computer vision, and token reduction has yielded promising outcomes in Vision Transformers (ViTs). However, token reduction performs less effectively on Vision Mamba compared to ViTs. Pruning informative tokens in Mamba leads to a high loss of key knowledge tokens and a drop in performance, making it not a good solution for enhancing efficiency. Token merging, which preserves more token information than pruning, has demonstrated commendable performance in ViTs, but vanilla merging performance decreases as the reduction ratio increases either, failing to maintain the key knowledge and performance in Mamba. Re-training the model with token merging, which effectively rebuilds the key knowledge, enhances the performance of Mamba. Empirically, pruned Vims, recovered on ImageNet-1K, only drop up to 0.9\% accuracy, by our proposed framework **R-MeeTo** in our main evaluation. We show how simple and effective the fast recovery can be achieved at minute-level, in particular, a 35.9\% accuracy spike over 3 epochs of training on Vim-Ti. Moreover, Vim-Ti/S/B are re-trained within 5/7/17 minutes, and Vim-S only drop 1.3\% with 1.2 $\times$ (up to 1.5 $\times$) speed up in inference.


## News 🚀
- `2024.12.12`: The code is released.


## 🛠 Dataset Prepare
- For image datasets we use ImageNet-1K.
- For video datasets K400, you can download them from [OpenDataLab](https://opendatalab.org.cn/OpenMMLab/Kinetics-400) or their official websites. We follow the data list from [here](https://drive.google.com/drive/folders/17VB-XdF3Kfr9ORmnGyXCxTMs86n0L4QL?usp=sharing) to split the dataset.

## 🛠 Installation

#### 1. Clone the repository

```
git clone https://github.com/Aristo23333/RMeeTo.git
```

#### 2. Create a new Conda environment

```
conda env create -f environment.yml
```
or install the necessary packages by requirement.txt

```
conda create -n R_MeeTo python=3.10.12
pip install -r requirements.txt
```
#### 3. Install Mamba package manually 

- For [Vim](https://github.com/hustvl/Vim) baseline: pip install the mamba package and casual-conv1d (version:1.1.1) in the Vim repo.
 ```
git clone https://github.com/hustvl/Vim
cd Vim 
pip install -e causal_conv1d==1.1.0
pip install -e mamba-1p1p1
```
- For [VideoMamba](https://github.com/OpenGVLab/VideoMamba) baseline: pip install the mamba package and casual-conv1d (version:1.1.0) in the VideoMamba repo.
 ```
git clone https://github.com/OpenGVLab/VideoMamba
cd VideoMamba
pip install -e causal_conv1d
pip install -e mamba
```

#### 4. Download the baseline pretrained models from our baseline official source
See [PRETRAINED](PRETRAINED.md) for downloading the pretrained model of our baseline. 


## ⚙️ Usage 

### 🛠️ Reproduce our results
#### For image task:
```
bash ./image_task/exp_sh/tab2/vim_tiny
```
#### For video task:
```
bash ./video_task/exp_sh/tab13/videomamba_tiny.sh
```
#### Checkpoints:

See [CKPT](CKPT.md) to find our reproduced checkpoints and logs of the main results. 

### ⏱️ Measure inference speed 
<p align="center">
<img src="./fig/speed.png" width=100% height=45%
class="center">

R-MeeTo effectively optimizes inference speed and is adaptable for both consumer-level, enterprise-level and other high-performance devices. See [this example](image_task/example/test_speed.ipynb) for testing FLOPS (G) and throughput (im/s).



### 🖼️ Visualization
<p align="center">
<img src="./fig/vis_exp.png" width=100% height=45%
class="center">

 See [this example](image_task/example/visualization.ipynb) of visualization of merged token on ImageNet-1k val using a re-trained Vim-S.

## Citation
If you found our work useful, please consider citing us.
```

```


## Acknowledge
The repo is partly built based on [ToMe](https://github.com/facebookresearch/ToMe), [Vision Mamba](https://github.com/hustvl/Vim), and [VideoMamba](https://github.com/OpenGVLab/VideoMamba). We are grateful for their generous contributions to open source.


## Contact
🔥🔥🔥 If you are interested in this work and hope to cooperate with us, please drop an email to yj1938@sjtu.edu.cn 🔥🔥🔥