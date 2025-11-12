## Create enviorment
```Shell
conda create -n matchstereo python=3.10
conda activate matchstereo
```

## For pytorch 2.5.1+cu124
```Shell
conda install -c "nvidia/label/cuda-12.4.0" cuda-toolkit
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
```

## For pytorch 2.7.1+cu128
```Shell
conda install nvidia/label/cuda-12.8.1::cuda-toolkit
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```

## 
```Shell
pip install imageio==2.9.0 imageio-ffmpeg==0.4.9 matplotlib==3.8.4 opencv-python==4.9.0.80 pillow==10.2.0 scikit-image==0.20.0 scipy==1.9.1 tensorboard==2.17.0 setuptools==59.5.0 psutil==6.0.0 joblib==1.4.2 numpy==1.24.4
pip install tqdm==4.66.2 timm==0.6.11
pip install deepspeed==0.14.4 # for flops profiler
pip install gradio==5.49.1 # for gradio demo
```

## (Optional) Install CUDA implementation of match attention
```Shell
cd models
bash compile.sh
```