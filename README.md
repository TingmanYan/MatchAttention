## ___***MatchAttention: Matching the Relative Positions for High-Resolution Cross-View Matching***___
<div align="center">
<img src='assets/logo.png' style="height:100px"></img>

 <a href='https://arxiv.org/abs/2510.14260'><img src='https://img.shields.io/badge/arXiv-2510.14260-b31b1b.svg'></a> &nbsp;
 <a href='https://matchattention.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
 <a href='https://huggingface.co/spaces/Tingman/MatchStereo'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a> &nbsp;

_**Tingman Yan, Tao Liu, Xilian Yang, Qunfei Zhao, Zeyang Xia**_
<br>
</div>

Please help ___***Star***___ this repo if you find it useful. Thank you!

## Introduction

MatchAttention is a contiguous and differentiable sliding-window attention mechanism that enables long-range connection, explict matching, and linear complexity. When applied to stereo matching and optical flow, real-time and state-of-the-arts performance can be achieved.

### FLOPs and memory consumption

<img src="assets/fig_acc_flops.svg" alt="match_attention" style="width:43%;"/> <img src="assets/fig_memory_resolution.svg" alt="match_attention" style="width:49%;"/>

### Zero-shot generalization

Strong zero-shot generalization on real-world datasets when trained on FSD Mix datasets.
<img src="assets/table_zero_shot.svg" alt="match_attention" style="width:100%;"/>
<br>
High-Resolution inference with fine-grained details
<img src="assets/high_res_details.png" alt="match_attention" style="width:100%;"/>
<br>
Real-time inference (MatchStereo-T @1280x720 on a RTX 4060 Ti GPU)

<img src="assets/realtime_demo.gif" alt="match_attention" style="width:100%;"/>

### Explainable occlusion handling

Top row shows the color image and GT occlusion mask from the Middlebury dataset (Playtable, 1852 x 2720).
Bottom row shows the cross relative position $R_{pos}[..., 0]$ (disparity) and the self relative position $sR_{pos}[..., 0]$ predicted by MatchStereo-B trained on FSD Mix datasets. 
The visualization of $sR_{pos}[..., 0]$ demonstrates that the attention sampling positions for occluded regions lie within their non-occluded neighboring regions.

<img src="assets/self_rpos_visualize.png" alt="self_rpos_visualize" style="width:100%;"/>

### Comparison with SOTA
MatchStereo-B ranked 1st in average error on the public [Middlebury benchmark](https://vision.middlebury.edu/stereo/eval3/) (2025-05-10)
<img src="assets/stereo_eval3.png" alt="Middlebury_leaderboard" style="width:100%;"/>
<br>
State-of-the-arts performance on four real-world benchmarks.
<img src="assets/benchmark_performance.png" alt="benchmark_performance" style="width:100%;"/>
## Model Weights
|Model|Params|Resolution|FLOPs|GPU Mem|Latency|Checkpoint|
|:---------|:---------|:--------|:--------|:--------|:--------|:--------|
|MatchStereo-T|8.78M|1536x1536|0.34T|1.45G|38ms|[Hugging Face](https://huggingface.co/Tingman/MatchAttention/blob/main/matchstereo_tiny_fsd.pth)|
|MatchStereo-S|25.2M|1536x1536|0.98T|1.73G|45ms|[Hugging Face](https://huggingface.co/Tingman/MatchAttention/blob/main/matchstereo_small_fsd.pth)|
|MatchStereo-B|75.5M|1536x1536|3.59T|2.94G|75ms|[Hugging Face](https://huggingface.co/Tingman/MatchAttention/blob/main/matchstereo_base_fsd.pth)|
|MatchFlow-B|75.5M|1536x1536|3.60T|3.22G|77ms|[Hugging Face](https://huggingface.co/Tingman/MatchAttention/blob/main/matchflow_base_sintel.pth)|
> GPU memory and latency measured on a single RTX 5090 GPU with torch.compile enabled and FP16 precision

## Setup
### 1. Clone MatchAttention
```Shell
git clone https://github.com/TingmanYan/MatchAttention
cd MatchAttention
```

### 2. Installation
```Shell
## Create enviorment
conda create -n matchstereo python=3.10
conda activate matchstereo
## For pytorch 2.5.1+cu124
conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit
conda install pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=12.4 -c pytorch -c nvidia
## For pytorch 2.7.1+cu128
conda install nvidia/label/cuda-12.8.1::cuda-toolkit
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
## other dependencies
pip install -r requirements.txt
## (Optional) Install CUDA implementation of match attention
cd models
bash compile.sh

## Download model weights to ./checkpoints
```
> [!NOTE]
> PyTorch 2.0+ is required for torch.compile

## Inference
### 1. Command line
```Shell
# on custom images
# stereo
python run_img.py --img0_dir images/left/ --img1_dir images/right/ --output_path outputs --checkpoint_path checkpoints/matchstereo_tiny_fsd.pth --no_compile
# flow
python run_img.py --img0_dir images/frame1/ --img1_dir images/frame2/ --output_path outputs --variant base --checkpoint_path checkpoints/matchflow_base_sintel.pth --mode flow --no_compile
# test on Middlebury
python run_img.py --middv3_dir images/MiddEval3/ --variant tiny --checkpoint_path checkpoints/matchstereo_tiny_fsd.pth --test_inference_time --inference_size 1536 1536 --mat_impl pytorch --precision fp16
python run_img.py --middv3_dir images/MiddEval3/ --variant small --checkpoint_path checkpoints/matchstereo_small_fsd.pth --test_inference_time --inference_size 2176 3840 --mat_impl cuda --precision fp16
python run_img.py --middv3_dir images/MiddEval3/ --variant base --checkpoint_path checkpoints/matchstereo_base_fsd.pth --mat_impl cuda --low_res_init --no_compile
# test on ETH3D
python run_img.py --eth3d_dir images/ETH3D/ --checkpoint_path checkpoints/matchstereo_tiny_fsd.pth --inference_size 416 832 --mat_impl pytorch --precision fp16 --device_id -1 # run on CPU
```

### 2. Local Gradio demo
```Shell
python gradio_app.py
```

### 3. Real-time inference using a ZED camera
```Shell
python zed_capture.py --checkpoint_path checkpoints/matchstereo_tiny_fsd.pth
```

## Citation
Please cite our paper if you find it useful
```
@article{yan2025matchattention,
  title={MatchAttention: Matching the Relative Positions for High-Resolution Cross-View Matching},
  author={Tingman Yan and Tao Liu and Xilian Yang and Qunfei Zhao and Zeyang Xia},
  journal={arXiv preprint arXiv:2510.14260},
  year={2025}
}
```

# Acknowledgement
We would like to thank the authors of [UniMatch](https://github.com/autonomousvision/unimatch), [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo), [MetaFormer](https://github.com/sail-sg/metaformer), and [TransNeXt](https://github.com/DaiShiResearch/TransNeXt) for their code release. Thanks to the author of [FoundationStereo](https://github.com/NVlabs/FoundationStereo) for the release of FSD dataset.

# Contact
Please reach out to [Tingman Yan](tingmanyan@dlut.edu.cn) for questions.