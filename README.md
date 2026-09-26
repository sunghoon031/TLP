# Robust Single-Parameter Point Cloud Registration using Truncated Lp Norms

This is the Python implementation of our work **"Robust Single-Parameter Point Cloud Registration using Truncated Lp Norms"**. 

The link to the paper will be provided here when it's ready.

It uses Pytorch for GPU parallelization (CUDA).

## Quick start:
1. Download the processed datasets [here](https://drive.google.com/drive/folders/1a17qggTB9dEfqe6kldd9xuPOtNKbcJFA?usp=sharing).
2. On terminal, type
````
conda env create -f environment.yml
conda activate TLP
````
3. To run, type
````
python evaluate.py
````

## Results on my laptop:
| Dataset        | Threshold used | mAA at 5 deg | mAA at 10 deg | mAA at 15 deg | Average time on my laptop (s) |
|----------------|----------------|--------------|---------------|---------------|-------------------------------|
| 3DMatch + FPFH | 0.09           | 0.587924     | 0.700246      | 0.745574      | 0.361329                      |
| 3DMatch + FCGF | 0.07           | 0.681824     | 0.804806      | 0.851715      | 0.199601                      |
| KITTI + FPFH   | 0.9            | 0.989910     | 0.994955      | 0.996637      | 0.652110                      |
| KITTI + FCGF   | 1.8            | 0.992432     | 0.994414      | 0.995075      | 0.395288                      |
