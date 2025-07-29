# DFWA-Net: Dual-domain Feature-enhanced with Wavelet Attention Network for SAR Ship Detection

## Installation

-Python 3.10

-PyTorch 1.12.1

-CUDA 11.3

-cuDNN 8.2.1

Building Gabor Conv

```bash
cd ultralytics/nn/modules
python setup.py build install 
```

## Running

Train the model:

```bash
python mytrain.py
```

## References

We would like to acknowledge the following works and their authors for inspiring and supporting our research:

@ARTICLE{10376280,
  author={Wang, Peng and Chen, Yongkang and Yang, Yi and Chen, Ping and Zhang, Gong and Zhu, Daiyin and Jie, Yongshi and Jiang, Cheng and Leung, Henry},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={A General Multiscale Pyramid Attention Module for Ship Detection in SAR Images}, 
  year={2024},
  volume={17},
  number={},
  pages={2815-2827},
  keywords={Feature extraction;Marine vehicles;Radar polarimetry;Semantics;Object detection;Data mining;Synthetic aperture radar;Feature map;multiscale pyramid attention;ship detection;synthetic aperture radar (SAR)},
  doi={10.1109/JSTARS.2023.3348269}}

@ARTICLE{11048866,
  author={Wang, Yuwu and Wu, Tieming and Guo, Limin and Mo, Yuhan},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Ship Target Detection in SAR Images Based on Multiple Attention Mechanism and Cross-Scale Feature Fusion}, 
  year={2025},
  volume={18},
  number={},
  pages={16517-16533},
  keywords={Marine vehicles;Radar polarimetry;Object detection;Accuracy;Computational modeling;Feature extraction;Target recognition;Optical distortion;Synthetic aperture radar;Robustness;Multiple attention;synthetic aperture radar (SAR) image;ship target detection},
  doi={10.1109/JSTARS.2025.3582861}}

@INPROCEEDINGS{10655712,
  author={Guo, Zhen and Gan, Hongping},
  booktitle={2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={CPP-Net: Embracing Multi-Scale Feature Fusion into Deep Unfolding CP-PPA Network for Compressive Sensing}, 
  year={2024},
  volume={},
  number={},
  pages={25086-25095},
  keywords={Deep learning;Fuses;Superresolution;Feature extraction;Distortion;Iterative methods;Image reconstruction;Compressive Sensing;CP-PPA;Deep Unfolding Networks},
  doi={10.1109/CVPR52733.2024.02370}}

@ARTICLE{11031212,
  author={Tang, Xiao and Cao, Kun and Xia, Yunzhi and Cui, Enkun and Zhao, Weining and Chen, Qiong},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={BESW-YOLO: A Lightweight SAR Image Detection Model Based on YOLOv8n for Complex Scenarios}, 
  year={2025},
  volume={18},
  number={},
  pages={16081-16094},
  keywords={Marine vehicles;Feature extraction;Computational modeling;Accuracy;Radar polarimetry;Convolution;Clutter;Attention mechanisms;YOLO;Noise;Deep learning;lightweight;ship detection;synthetic aperture radar (SAR);YOLOv8},
  doi={10.1109/JSTARS.2025.3579292}}

@ARTICLE{10365386,
  author={Hao, Yisheng and Zhang, Ying},
  journal={IEEE Transactions on Aerospace and Electronic Systems}, 
  title={A Lightweight Convolutional Neural Network for Ship Target Detection in SAR Images}, 
  year={2024},
  volume={60},
  number={2},
  pages={1882-1898},
  keywords={Object detection;Radar polarimetry;Marine vehicles;Classification algorithms;Feature extraction;Convolutional neural networks;Clutter;Anchor free;lightweight convolutional neural network;synthetic aperture radar (SAR);target detection},
  doi={10.1109/TAES.2023.3344396}}

@ARTICLE{10365386,
  author={Hao, Yisheng and Zhang, Ying},
  journal={IEEE Transactions on Aerospace and Electronic Systems}, 
  title={A Lightweight Convolutional Neural Network for Ship Target Detection in SAR Images}, 
  year={2024},
  volume={60},
  number={2},
  pages={1882-1898},
  keywords={Object detection;Radar polarimetry;Marine vehicles;Classification algorithms;Feature extraction;Convolutional neural networks;Clutter;Anchor free;lightweight convolutional neural network;synthetic aperture radar (SAR);target detection},
  doi={10.1109/TAES.2023.3344396}}

