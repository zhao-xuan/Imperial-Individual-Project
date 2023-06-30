# Dataset prepration

* `crop.ipynb`: Crop images from the LUNA16 dataset and the diffusion model generated images, and produce `labels.csv` and `labels_all.csv`.
* `mmcls.ipynb`: Samples from `labels.csv`/`labels_all.csv` and generates annotation files (`train_ann.txt`/`train_gen_ann.txt`/`test_ann.txt`) for consumption by MMClassification.
<!-- 
## Steps to prepare for the local-global model training

1. Run `crop.ipynb` to get an `output` directory.
2. Apply patches (`local_global_patches/0001-update.patch` and `local_global_patches/0002-update.patch`) sequentially to the original local-global repo. (with `git am < PATH_TO_THE_PATCH` under your local-global repo).
3. Run the local-global model training script. Specify the `output` directory as the dataset location.
4. If training with diffusion-generated images, set `use_generated_nodules` to true in `preprocessing.py` under your local-global repo root. Otherwise, set it to false. -->

## Steps to prepare for MMClassification training

1. Run `crop.ipynb` to get an `output` directory, if it doesn't exist.
2. Run `mmcls.ipynb` to generate annotation files, which will be under the `output` directory.
3. Copy `mmclassification_configs/new_configs.py` to `configs` under your MMClassification repo root.
4. Modify `configs/new_config.py` in the MMClassification directory. Set `data_prefix` to the location of the `output` directory.
    * If we want to include images from the diffusion model, set `data.train.ann_file` to `data_prefix + "/train_gen_ann.txt"`
    * if we want to exclude images from the diffusion model, set `data.train.ann_file` to `data_prefix + "/train_ann.txt"`
5. Run the MMClassification training script, specifiying `configs/new_config.py` as the config.

