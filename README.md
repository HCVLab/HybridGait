# HybridGait

<!-- Project Page | Video | Paper | Data -->

<!-- ![Reconstructed Objects](path_to_images) -->

**Association for the Advancement of Artificial Intelligence(AAAI), 2024**

**Authors: Yilan Dong,Chunlin Yu,Ruiyang Ha,Ye Shi,Yuexin Ma,Lan Xu,Yanwei Fu,Jingya Wang**

**Paper:** https://arxiv.org/abs/2401.00271

<!-- Conference: AAAI 2024 -->

<!-- This is the official repo for the implementation of HybridGait: A Benchmark for Spatial-Temporal Cloth-Changing Gait Recognition with Hybrid Explorations -->

<!-- ## Updates

- 2023/06/19: The hand tracking code is released here: [EasyMocap](link_to_EasyMocap)
- 2023/02/13: For people who do not want to run hand tracking, we provide the processed hand tracking results: [HOD_S1_HT](link) and [HOD_D1_HT](link).  -->

## Abstract
Existing gait recognition benchmarks mostly include minor clothing variations in the laboratory environments, but lack persistent changes in appearance over time and space. In this paper, we propose the first in-the-wild benchmark CCGait for cloth-changing gait recognition, which incorporates di- verse clothing changes, indoor and outdoor scenes, and multi- modal statistics over 92 days. To further address the cou- pling effect of clothing and viewpoint variations, we pro- pose a hybrid approach HybridGait that exploits both tem- poral dynamics and the projected 2D information of 3D hu- man meshes. Specifically, we introduce a Canonical Align- ment Spatial-Temporal Transformer (CA-STT) module to en- code human joint position-aware features, and fully exploit 3D dense priors via a Silhouette-guided Deformation with 3D-2D Appearance Projection (SilD) strategy. Our contri- butions are twofold: we provide a challenging benchmark CCGait that captures realistic appearance changes across an expanded and space, and we propose a hybrid frame- work HybridGait that outperforms prior works on CCGait and Gait3D benchmarks.

## Dataset Download

Please fullfile the [aggrement](https://drive.google.com/file/d/1X7f7u_ddjadQllaTyTaRb6E0kw5uTwof/view?usp=drive_link) to get the CCGait Dataset. 

You can download the Gait3D rendered front view silhouette(SMPL model) from [here](https://drive.google.com/drive/folders/13VmVJ-l_ybCu0sOmYXx0g5m_ANOUXIuH?usp=sharing).

## TODO List

- [x] Release the CCGait dataset.
- [x] Release the Rendered front view data.

## Installation

<!-- ### Set up the environment -->
```shell

git clone https://github.com/HCVLab/HybridGait.git
conda create -n HybridGait python=3.7
conda activate HybridGait
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111  -f https://download.pytorch.org/whl/torch_stable.html
cd HybridGait
pip install -r requirement.txt

```

## Checkpoints
Coming soon

## Running

- **Training**
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 60496 lib/main.py --cfgs ./config/HybridGait_64pixel.yaml --phase train
```

- **Testing**
```shell
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 37584 lib/main.py --cfgs ./config/HybridGait_64pixel.yaml --phase test
```


## Citation

Cite as below if you find this repository is helpful to your project:

```
@article{dong2023hybridgait,
  title={HybridGait: A Benchmark for Spatial-Temporal Cloth-Changing Gait Recognition with Hybrid Explorations},
  author={Dong, Yilan and Yu, Chunlin and Ha, Ruiyang and Shi, Ye and Ma, Yuexin and Xu, Lan and Fu, Yanwei and Wang, Jingya},
  journal={arXiv preprint arXiv:2401.00271},
  year={2023}
}
```