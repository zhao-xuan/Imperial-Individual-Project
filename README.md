# Imperial College London Department of Computing Year 4 Individual Project

## Project Title: High-Fidelity Image Synthesis from Pulmonary Nodule Lesion Maps using Semantic Diffusion Model

### Abstract
Lung cancer has been one of the leading causes of cancer-related deaths worldwide for years. With the emergence of deep learning, computer-assisted diagnosis (CAD) models based on learning algorithms can accelerate the nodule screening process, providing valuable assistance to radiologists in their daily clinical workflows. However, developing such robust and accurate models often requires large-scale and diverse medical datasets with high-quality annotations. Generating synthetic data provides a pathway for augmenting datasets at a larger scale. Therefore, in this paper, we explore the use of Semantic Diffusion Models (SDM) to generate high-fidelity pulmonary CT images from segmentation maps. We utilize annotation information from the LUNA16 dataset to create paired CT images and masks, and assess the quality of the generated images using the Fr ́echet Inception Distance (FID), as well as on two common clinical downstream tasks: nodule detection and nodule localization. Achieving improvements of 3.953% for detection accuracy and 8.5% for AP50 in nodule localization task, respectively, demonstrates the feasibility of the approach.

### Project Structure
```
├── README.md
├── data
│   ├── LUNA16
│   │   ├── annotations
│   │   ├── images
│   │   └── masks
│   ├── LUNA16_2D
│   │   ├── annotations
│   │   ├── images
│   │   └── masks
│   ├── LUNA16_3D
│   │   ├── annotations
│   │   ├── images
│   │   └── masks
│   ├── LUNA16_3D_2
│   │   ├── annotations
│   │   ├── images
│   │   └── masks
│   ├── LUNA16_3D_3
│   │   ├── annotations
│   │   ├── images
