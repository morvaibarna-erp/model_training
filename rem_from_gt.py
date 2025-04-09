

import os

# Folder containing images and the gt.txt file
folder_path = "./new_ocr_dataset/"
gt_file_path = os.path.join(folder_path, "gt.txt")

# Get the list of image files in the folder
image_files = set(os.listdir(folder_path))

# Open the gt.txt file and read all lines
with open(gt_file_path, "r") as gt_file:
    lines = gt_file.readlines()

# Filter out lines where the corresponding image file does not exist
updated_lines = []
for line in lines:
    # Split the line to extract the image file name (before the tab character)
    image_file = line.split('\t')[0]
    
    # Check if the image file exists in the folder
    if image_file in image_files:
        updated_lines.append(line)

# Write the updated lines back to gt.txt
with open(gt_file_path, "w") as gt_file:
    gt_file.writelines(updated_lines)

print(f"Filtered gt.txt file successfully. Removed missing images from the list.")
