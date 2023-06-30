import numpy as np
import pandas as pd
from tqdm import tqdm

import SimpleITK as sitk
import matplotlib.pyplot as plt

import skimage.io
import skimage.transform
from pathlib import Path

root = Path("/vol/bitbucket/bh1511/data/LUNA16")
output_dir = Path("./luna16_cropped")
output_dir.mkdir(exist_ok=True, parents=True)

candidates_df = pd.read_csv(root / "candidates_V2.csv")

def load_image(file):
    image_itk = sitk.ReadImage(file)
    image_itk = sitk.IntensityWindowing(image_itk, 
                                        windowMinimum=-1000, windowMaximum=400, 
                                        outputMinimum=0.0, outputMaximum=255.0)
    image_itk = sitk.Cast(image_itk, sitk.sitkUInt8)
    image_arr = sitk.GetArrayFromImage(image_itk)
    origin = np.array(list(image_itk.GetOrigin()))
    space = np.array(list(image_itk.GetSpacing()))
    return image_arr, origin, space
  
  
# First add a vZ column in the average_malignancy.csv
# Then run the following code to generate the labels_all_with_malignancy.csv
image_missing_candidate_indices = []
avg_malignancy_df = pd.read_csv("./average_malignancy.csv")
vZ = []

for i, row in avg_malignancy_df.iterrows():
  print("Processing row: ", i, " out of ", len(avg_malignancy_df), " rows")
  for series_uid, series_candidates in candidates_df.groupby("seriesuid"):
    if series_uid == row.seriesuid.split("_")[0]:
      try:
          image, origin, space = load_image(root / f"images/{series_uid}.mhd")
      except:
          print(f"Image for {series_uid} does not exist, skipping")
          image_missing_candidate_indices += list(series_candidates.index)
          continue

      series_candidates = series_candidates[series_candidates["class"] == 1]

      for i, candidate in series_candidates.iterrows():
        if abs(float(candidate["coordZ"]) - float(row['z_coord'])) > 2:
          continue
        node_x = candidate["coordX"]     # X coordinate of the nodule
        node_y = candidate["coordY"]     # Y coordinate of the nodule
        node_z = candidate["coordZ"]     # Z coordinate of the nodule

        # nodule center (x,y,z ordering)
        center = np.array([node_x, node_y, node_z])
        # nodule center in voxel space (x,y,z ordering)
        v_center = np.rint((center - origin) / space).astype('int')

        v_x, v_y, v_z = v_center
        # add the vZ field for the average_malignancy.csv
  
  # avg_malignancy_df.loc[i, "vZ"] = v_z
  vZ.append(v_z)
  
print(vZ)

# Write the average_malignancy.csv with the vZ field
avg_malignancy_df.to_csv("./average_malignancy_with_vZ.csv", index=False)