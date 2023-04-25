# Imperial College London Department of Computing Year 4 Individual Project

## Project Title: High-Fidelity Image Synthesis from Pulmonary Nodule Lesion Maps using Semantic Diffusion Model

### Abstract
Lung cancer has been one of the leading causes of cancer-related deaths worldwide for years. With the emergence of deep learning, computer-assisted diagnosis (CAD) models based on learning algorithms can accelerate the nodule screening process, providing valuable assistance to radiologists in their daily clinical workflows. However, developing such robust and accurate models often requires large-scale and diverse medical datasets with high-quality annotations. Generating synthetic data provides a pathway for augmenting datasets at a larger scale. Therefore, in this paper, we explore the use of Semantic Diffusion Models (SDM) to generate high-fidelity pulmonary CT images from segmentation maps. We utilize annotation information from the LUNA16 dataset to create paired CT images and masks, and assess the quality of the generated images using the Fr ́echet Inception Distance (FID), as well as on two common clinical downstream tasks: nodule detection and nodule localization. Achieving improvements of 3.953% for detection accuracy and 8.5% for AP50 in nodule localization task, respectively, demonstrates the feasibility of the approach.

### Project Structure
```
├── README.md
├── lung-cancer-detection.ipynb: an exploratory LUNA16 dataset preprocessing notebook I found [online](https://github.com/ayush9304/Lung_Cancer_Detection/blob/main/notebooks/v2/01_Lungs%20ROI%20_%20Nodule%20Mask%20extraction%20from%20LUNA16%20dataset.ipynb).
├── lunda16-preprocess-from-benjamin.ipynb: use this notebook to extract the LUNA16 dataset into the correct format for the SDM.
├── luna16-preprocess.ipynb: another exploratory LUNA16 dataset preprocessing notebook I found [online](https://github.com/s-mostafa-a/Luna16/blob/master/notebooks/Preprocessor.ipynb)
├── luna16-generate-seg-map.ipynb: modified version of luna16-preprocess.ipynb, basically just a copy.
├── experiment-to-visualize-luna16.ipynb: an exploratory notebook to visualize and overlay nodules onto the lung.
├── ddim-keras-example.ipynb: implementation of [this DDIM Keras example](https://keras.io/examples/generative/ddim/).
├── ddim-keras-example.py: python version of ddim-keras-example.ipynb.
├── imgs_oxford_flowers: folder containing the diffusion images generated based on the Oxford Flowers dataset.
├── SDM: modified version of [Semantic Diffusion Model](https://github.com/WeilunWang/semantic-diffusion-model).
│   ├── assets
│   ├── evaluations
│   ├── guided_diffusion
│   │   ├── guassian_diffusion.py
│   │   └── image_datasets.py
│   ├── scripts
│   │   └── luna16.sh
│   ├── image_sample.py
│   └── image_train.py
│   
├── NoduleGAN
│   ├── data
│   │   └── process_LUNA16.ipynb
│   ├── run_pipeline.py: for generating healthy synthetic GAN scans.
│   └── eval_gaugan.py: for generating nodule synthetic GAN scans.
│
├── data