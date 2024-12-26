PMCE++ extends the original [PMCE](https://github.com/kasvii/PMCE), presented at ICCV 2023, to achieve pose-and-shape accuracy, temporal consistency, and efficiency in 3D human mesh reconstruction from videos.

<p align="center">
    <img src="assets/framework.png" /> 
</p>
<p align="center">
  <img src="assets/demo1.gif" height="110" /> 
  <img src="assets/demo2.gif" height="110" /> 
  <img src="assets/demo3.gif" height="110" /> 
  <img src="assets/demo4.gif" height="110" /> 
</p>

## Preparation

1. Install dependencies. This project is developed on Ubuntu 18.04 with NVIDIA 3090 GPUs. We recommend you to use an [Anaconda](https://www.anaconda.com/) virtual environment.
```bash
# Create a conda environment.
conda create -n pmce_plus python=3.9
conda activate pmce_plus

# Install PyTorch >= 1.2 according to your GPU driver.
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -c conda-forge

# Pull the code
git clone https://github.com/kasvii/PMCE-plus.git --recursive
cd PMCE-plus

# Install other dependencies.
pip install -r requirements.txt

# Install ViTPose
pip install -v -e third-party/ViTPose

# Install VirtualPose
pip install -v -e third-party/VirtualPose
```


## Quick Demo
```bash
python demo.py --video examples/demo.mp4 --visualize
```

## Test
```bash
python -m lib.eval.evaluate_3dpw --cfg configs/yamls/test_3dpw.yaml
```

```bash
python -m lib.eval.evaluate_h36m --cfg configs/yamls/test_h36m.yaml
```

```bash
python -m lib.eval.evaluate_mpii3d --cfg configs/yamls/test_mpii3d.yaml
```

## Train
```bash
python train.py --cfg configs/yamls/stage2.yaml --gpu 0
```