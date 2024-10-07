import os

# Specify the folder path
folder_path = 'dataset/train/images'

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    # Check if the file has a .JPG extension
    if filename.endswith('.JPG'):
        # Construct the full file path
        old_file = os.path.join(folder_path, filename)
        # Construct the new file path with the .jpg extension
        new_file = os.path.join(folder_path, filename.replace('.JPG', '.jpg'))
        # Rename the file
        os.rename(old_file, new_file)
        print(f'Renamed: {old_file} -> {new_file}')
