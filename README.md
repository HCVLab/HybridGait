# HybridGait

<!-- Project Page | Video | Paper | Data -->

<!-- ![Reconstructed Objects](path_to_images) -->

HybridGait: A Benchmark for Spatial-Temporal Cloth-Changing Gait Recognition with Hybrid Explorations

Authors: Yilan Dong,Chunlin Yu,Ruiyang Ha,Ye Shi,Yuexin Ma,Lan Xu,Yanwei Fu,Jingya Wang
Conference: AAAI 2024

<!-- This is the official repo for the implementation of HybridGait: A Benchmark for Spatial-Temporal Cloth-Changing Gait Recognition with Hybrid Explorations -->

<!-- ## Updates

- 2023/06/19: The hand tracking code is released here: [EasyMocap](link_to_EasyMocap)
- 2023/02/13: For people who do not want to run hand tracking, we provide the processed hand tracking results: [HOD_S1_HT](link) and [HOD_D1_HT](link).  -->

## TODO List

- [ ] Release the CCGait dataset.
- [ ] Release the Rendered front view data.

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