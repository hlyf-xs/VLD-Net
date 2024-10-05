# VLD-Net
This is the official pytorch implementation of "VLD-Net: Localization and Detection of the Vertebrae from X-ray Images by Reinforcement Learning with Adaptive Exploration Mechanism and Spine Anatomy Information"

## Requirements
CUDA 11.4<br />
Python 3.7<br /> 
Tensorflow-gpu 2.5.0<br /> 
Tensorboard 2.11.0<br />


## Usage

### Installation
* Clone this repo
```
git clone https://github.com/hlyf-xs/VLD-Net.git
cd VLD-Net
```

2. Download [MICCAI 2019 AASCE Challenge Dataset](https://aasce19.github.io/), and [BUU LSPINE dataset](https://services.informatics.buu.ac.th/spine/), then put them into `data/`

3. Train the model
```
sh run.sh
```



# Citation

If you find our work is useful for you, please cite us.

### Contact
Shun Xiang (XiangShun1997@gmail.com)