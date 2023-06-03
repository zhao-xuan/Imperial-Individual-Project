import cv2
import os
import numpy as np

def get_image_min_max(folder_path):
    image_files = os.listdir(folder_path)
    if not image_files:
        print("No images found in the folder.")
        return None

    # Initialize min and max values
    min_value = float('inf')
    max_value = float('-inf')

    for file_name in image_files:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            # Load the image
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            if image is not None:
                # Calculate min and max values
                image_min = np.min(image)
                image_max = np.max(image)

                # Update overall min and max values
                if image_min < min_value:
                    min_value = image_min
                if image_max > max_value:
                    max_value = image_max
        else:
            print(f"Ignoring non-image file: {file_name}")

    return min_value, max_value

# Example usage
folder_path = '/home/user/conditional/nodule_labels_ohe'  # Replace with the actual folder path
min_val, max_val = get_image_min_max(folder_path)

if min_val is not None and max_val is not None:
    print(f"Minimum value: {min_val}")
    print(f"Maximum value: {max_val}")
