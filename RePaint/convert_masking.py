import os
import cv2

def process_images(directory):
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Specify the image formats you want to process
            # Read the image
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            # Perform pixel manipulation
            image[image == 0] = 255
            image[image == 1] = 0
            
            # Save the modified image
            modified_filename = f"modified_{filename}"
            modified_image_path = os.path.join(directory, modified_filename)
            cv2.imwrite(modified_image_path, image)
            print(f"Processed image saved as {modified_filename}")

# Provide the directory path where the images are located
image_directory = "/home/user/luna-16-seg-diff-data/only_nodule_labels_ohe"
process_images(image_directory)