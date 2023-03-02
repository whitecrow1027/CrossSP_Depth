# Cross-spectrum Depth Estimation

Cross-Spectrum Unsupervised Depth Estimation by Visible-light and Thermal Cameras

[papre link]

## Training
0. requriment:

    pytorch >= 1.7.0

    cuda >= 10.0

    python >= 3.6

1. Prepare dataset

    Download [VTD dataset](https://github.com/whitecrow1027/VIS-TIR-Datasets).

2. Edit train.sh to add your dataset path.

    `CUDA_VISIBLE_DEVICES=0 python train.py --mscroot yourpath`

    then, run train.sh.
