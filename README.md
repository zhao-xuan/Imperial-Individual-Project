# Imperial College London Department of Computing Year 4 Individual Project

## Project Title: High-Fidelity Image Synthesis from Pulmonary Nodule Lesion Maps using Semantic Diffusion Model

### Abstract
Lung cancer has been one of the leading causes of cancer-related deaths worldwide for years. With the emergence of deep learning, computer-assisted diagnosis (CAD) models based on learning algorithms can accelerate the nodule screening process, providing valuable assistance to radiologists in their daily clinical workflows. However, developing such robust and accurate models often requires large-scale and diverse medical datasets with high-quality annotations. Generating synthetic data provides a pathway for augmenting datasets at a larger scale. Therefore, in this paper, we explore the use of Semantic Diffusion Models (SDM) to generate high-fidelity pulmonary CT images from segmentation maps. We utilize annotation information from the LUNA16 dataset to create paired CT images and masks, and assess the quality of the generated images using the Fr ́echet Inception Distance (FID), as well as on two common clinical downstream tasks: nodule detection and nodule localization. Achieving improvements of 3.953% for detection accuracy and 8.5% for AP50 in nodule localization task, respectively, demonstrates the feasibility of the approach.

### Project Structure
<pre>
├── README.md
├── lung-cancer-detection.ipynb: an exploratory LUNA16 dataset preprocessing notebook I found <a href="https://github.com/ayush9304/Lung_Cancer_Detection/blob/main/notebooks/v2/01_Lungs%20ROI%20_%20Nodule%20Mask%20extraction%20from%20LUNA16%20dataset.ipynb">online</a>.
├── lunda16-preprocess-from-benjamin.ipynb: use this notebook to extract the LUNA16 dataset into the correct format for the SDM.
├── luna16-preprocess.ipynb: another exploratory LUNA16 dataset preprocessing notebook I found <a href="https://github.com/s-mostafa-a/Luna16/blob/master/notebooks/Preprocessor.ipynb">online</a>
├── luna16-generate-seg-map.ipynb: modified version of luna16-preprocess.ipynb, basically just a copy.
├── experiment-to-visualize-luna16.ipynb: an exploratory notebook to visualize and overlay nodules onto the lung.
├── ddim-keras-example.ipynb: implementation of <a href="https://keras.io/examples/generative/ddim/">this DDIM Keras example</a>.
├── ddim-keras-example.py: python version of ddim-keras-example.ipynb.
├── imgs_oxford_flowers: folder containing the diffusion images generated based on the Oxford Flowers dataset.
├── SDM: modified version of <a href="https://github.com/WeilunWang/semantic-diffusion-model">Semantic Diffusion Model</a>.
│   ├── assets
│   ├── evaluations
│   ├── guided_diffusion: main folder for the SDM.
│   │   ├── guassian_diffusion.py: diffusion model and schedule setup.
│   │   └── image_datasets.py: modified in order to load LUNA16 dataset.
│   ├── scripts
│   │   └── luna16.sh: modified in order to train an SDM on LUNA16 dataset.
│   ├── image_sample.py: inference script for SDM.
│   └── image_train.py: training script for SDM.
│   
├── NoduleGAN: a GauGAN-based generative model for generating synthetic lung CT scans, used for comparison against SDM performance.
│   ├── data
│   │   └── process_LUNA16.ipynb
│   ├── run_pipeline.py: for generating healthy synthetic GAN scans.
│   └── eval_gaugan.py: for generating nodule synthetic GAN scans.
│
├── mmclassification: modified version of [mmclassification](https://github.com/open-mmlab/mmpretrain) (Now re-named to mmpretrain)
│   └── configs
│       └── new_config.py: custom config file, required to train classification model.
└── mmdetection: modified version of [mmclassification](https://github.com/open-mmlab/mmdetection)
    └── configs
        └── *.py: all custom config files, required to train detection model.
</pre>

### References
- This project is largely based on the [Semantic Diffusion Model](https://github.com/WeilunWang/semantic-diffusion-model)
- This project is using the [LUNA16 dataset](https://luna16.grand-challenge.org/)
- This project is partially re-implementing [this paper](https://www.sciencedirect.com/science/article/pii/S1361841522001384)
- The project report is [here](https://www.overleaf.com/read/xtfshtxhnwdg)