# Reproducible Image Segmentation Experiment

## Overview
This repository contains the code and evaluation scripts for a controlled image segmentation experiment. The objective is to evaluate the impact of a single architectural mechanism—modifying the bottleneck layer of a modest U-Net to use **dilated (atrous) convolutions**—on the network's ability to segment pedestrians in urban scenes.

By strictly controlling the training budget and parameter count, this project isolates the effect of expanding the receptive field on the Intersection-over-Union (IoU) metric.

## Dataset
* **Name:** [Penn-Fudan Pedestrian Dataset](https://www.cis.upenn.edu/~jshi/ped_html/)
* **Classes:** 
  * `Class 0`: Background
  * `Class 1`: Pedestrian
* **Preprocessing:** Images are resized to `128x128`, normalized using ImageNet RGB statistics, and converted to PyTorch Tensors.

## Architecture & Controlled Change
The base architecture is a modest U-Net (Encoder-Decoder) with 117,058 parameters.
* **Baseline Model:** Utilizes standard 3x3 convolutions in the central bottleneck block.
* **Experimental Model:** Replaces the standard bottleneck convolutions with **Dilated Convolutions** (`dilation=2`, `padding=2`) to expand the network's receptive field without adding any extra learnable parameters.

### Training Budget constraints
* **Stopping Rule:** Strictly 20 epochs.
* **Optimizer:** Adam (lr = 1e-3).
* **Loss Function:** CrossEntropyLoss with class weights (`Background=0.2`, `Pedestrian=0.8`) to counter severe class imbalance.
* **Seed:** 42 (for complete reproducibility).

## Results
The models were evaluated on a Google Colab instance (NVIDIA T4 GPU). The dilated bottleneck successfully increased the mean IoU by expanding context aggregation, all without inflating the parameter count or inference latency.

| Model | Parameters | Inference Latency | Pedestrian IoU | Mean IoU (mIoU) |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline (Standard)** | 117,058 | ~0.33s | 0.3431 | **0.5486** |
| **Dilated Bottleneck** | 117,058 | ~0.32s | 0.4205 | **0.6351** |

*Note: Due to the modest architecture, strict 20-epoch budget, and heavy pedestrian class weighting, the models heavily overestimate pedestrian boundaries ("blobbing"). This experiment proves the isolated theoretical advantage of dilation, but does not establish production-ready accuracy.*

## Repository Structure
* `Assignment_B.ipynb`: The complete, reproducible Google Colab notebook containing the data contract, model architectures, training loop, evaluation script, and visualization generation.
* `23115064_Report.pdf`: The formal project report containing architectural diagrams and in-depth failure analysis (submitted separately).

## How to Run
1. Open the `.ipynb` notebook in Google Colab.
2. Ensure the Runtime is set to a GPU (T4 recommended).
3. Run all cells sequentially. The script will automatically download the dataset, train both models, print the evaluation metrics, and plot the visual evidence.
